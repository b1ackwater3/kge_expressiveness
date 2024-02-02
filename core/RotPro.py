

import torch
import torch.nn as nn

def modify_grad(x, inds):
    x[inds] = 0 
    return x

class RotPro(nn.Module):

    def __init__(self,n_entity, n_relation, dim, train_pr_prop=1, gamma=None):
        super(RotPro,self).__init__()
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.train_pr_prop = train_pr_prop
        self.epsilon = 2.0
        self.entity_dim = dim*2 
        self.relation_dim = dim
        
        self.entity_embedding = nn.Embedding(n_entity,self.entity_dim)
        self.relation_embedding = nn.Embedding(n_relation,self.relation_dim)

        self.projection_embedding_a = nn.Embedding(n_relation, self.relation_dim)
        self.projection_embedding_b = nn.Embedding(n_relation, self.relation_dim)
        self.projection_phase = nn.Embedding(n_relation, self.relation_dim) 
        nn.init.uniform_(
            tensor=self.projection_phase.weight, 
            a=0.75, 
            b=0.75
        )
        self.type_trans = nn.Linear(self.entity_dim,self.entity_dim)

        if gamma != None:
            self.embedding_range = (2 + gamma)/dim
            nn.init.uniform_(
                tensor=self.entity_embedding.weight, 
                a=-self.embedding_range, 
                b=self.embedding_range
            )
            nn.init.uniform_(
                tensor=self.relation_embedding.weight, 
                a=-self.embedding_range, 
                b=self.embedding_range
            )
        else:
            self.embedding_range = (50)/dim
            nn.init.xavier_uniform_(self.entity_embedding.weight)
            nn.init.xavier_uniform_(self.relation_embedding.weight)

        nn.init.uniform_(
                tensor=self.projection_embedding_a.weight,
                a=0.5,
                b=0.5
            )
        # nn.init.uniform_(
        #         tensor=self.projection_embedding_b.weight,
        #         a=0.5,
        #         b=0.5
        # )
        nn.init.uniform_(
                tensor=self.projection_embedding_b.weight,
                a=1,
                b=1
        )
        self.projection_embedding_b.weight.detach_()
        
    def score_function(self, head, relations, tail):
        relation = relations[0]
        proj_a = relations[1]
        proj_b = relations[2]
        proj_p = relations[3]
        pi = 3.14159265358979323846

        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail,im_tail = torch.chunk(tail, 2, dim=2)  

        # Make phases of relations uniformly distributed in [-pi, pi]
        phase_relation = (relation / (self.embedding_range / pi) )  * self.train_pr_prop
        phase_projection = proj_p

        re_projection = torch.cos(phase_projection)
        im_projection = torch.sin(phase_projection)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        ma = re_projection * re_projection * proj_a + im_projection * im_projection * proj_b
        mb = re_projection * im_projection * (proj_b - proj_a)
        md = re_projection * re_projection * proj_b + im_projection * im_projection * proj_a
        # p(et)
        re_tail_proj = ma * re_tail + mb * im_tail
        im_tail_proj = mb * re_tail + md * im_tail

        # p(eh)
        re_head_proj = ma * re_head + mb * im_head
        im_head_proj = mb * re_head + md * im_head

        if tail.shape[1] == relation.shape[1]:
            re_score = re_relation * re_tail_proj + im_relation * im_tail_proj
            im_score = re_relation * im_tail_proj - im_relation * re_tail_proj
            re_score = re_score - re_head_proj
            im_score = im_score - im_head_proj

        else:
            re_score = re_head_proj * re_relation - im_head_proj * im_relation
            im_score = re_head_proj * im_relation + im_head_proj * re_relation
            re_score = re_score - re_tail_proj
            im_score = im_score - im_tail_proj

        score = torch.stack([re_score, im_score],dim=0)  
        score = score.norm(dim=0)
        score = - score.sum(dim=2)
        return score

    def forward(self, h,r,t, mode='hrt',type_trans=False):
        head = None
        tail = None
        relation = self.relation_embedding(r).unsqueeze(1)
        proj_a = self.projection_embedding_a(r).unsqueeze(1)
        proj_b = self.projection_embedding_b(r).unsqueeze(1)
        proj_p = self.projection_phase(r).unsqueeze(1)
        if mode=='hr_t':
            negative_size = t.shape[1]
            batch_size  = t.shape[0]
            t = t.reshape(-1,1)
            head = self.entity_embedding(h).unsqueeze(1)
            tail = self.entity_embedding(t).reshape(batch_size,negative_size,-1)
        elif mode == 'h_rt':
            negative_size = h.shape[1]
            batch_size  = h.shape[0]
            h = h.reshape(-1,1)
            head = self.entity_embedding(h).reshape(batch_size,negative_size,-1)
            tail = self.entity_embedding(t).unsqueeze(1)         
        elif mode == 'hrt':
            head = self.entity_embedding(h).unsqueeze(1)
            tail = self.entity_embedding(t).unsqueeze(1)     

        if type_trans:
            tail = self.type_trans(tail)

        relations = [ relation,proj_a,proj_b,proj_p ]
        return self.score_function(head, relations,tail)


    def caculate_constarin(self,gamma_m,beta,alpha):
        a1 = self.projection_embedding_a.weight - 1.0
        a0 = self.projection_embedding_a.weight - 0.0
        a = torch.abs(a1 * a0)
        penalty = torch.ones_like(a)
        penalty[a > gamma_m] = beta
        l_a = (a * penalty).norm(p=2)
        b1 = self.projection_embedding_b.weight - 1.0
        b0 = self.projection_embedding_b.weight - 0.0
        b = torch.abs(b1 * b0)
        penalty = torch.ones_like(b)
        penalty[b > gamma_m] = beta
        l_b = (b * penalty).norm(p=2)
        loss = (l_a + l_b) * alpha
        return loss

    # exp8 
    # def caculate_constarin(self,gamma_m,beta,alpha):
    #     a1 = self.projection_embedding_a.weight - 1.0
    #     a0 = self.projection_embedding_a.weight - 0.0
    #     a = torch.abs(a1 * a0)
    #     penalty = torch.ones_like(a)
    #     penalty[a > gamma_m] =  (gamma_m - a[a > gamma_m])  * beta
    #     l_a = (a * penalty).norm(p=2)
    #     b1 = self.projection_embedding_b.weight - 1.0
    #     b0 = self.projection_embedding_b.weight - 0.0
    #     b = torch.abs(b1 * b0)
    #     penalty = torch.ones_like(b)
    #     penalty[b > gamma_m] =  (gamma_m - b[b > gamma_m])  * beta
    #     l_b = (b * penalty).norm(p=2)
    #     loss = (-l_a - l_b) * alpha
    #     return loss

    # exp6
    # def caculate_constarin(self,gamma_m,beta,alpha):
    #     epsilon =  0.05
    #     a1 = self.projection_embedding_a.weight - 1.0
    #     a0 = self.projection_embedding_a.weight - 0.0
    #     a = torch.abs(a1 * a0)
    #     penalty = torch.ones_like(a)
    #     penalty[a > gamma_m] =  1.0/(a[a > gamma_m]* beta + epsilon)
    #     l_a = (a * penalty).norm(p=2)
    #     b1 = self.projection_embedding_b.weight - 1.0
    #     b0 = self.projection_embedding_b.weight - 0.0
    #     b = torch.abs(b1 * b0)
    #     penalty = torch.ones_like(b)
    #     penalty[b > gamma_m] =   1.0/(b[b > gamma_m]* beta + epsilon)  
    #     l_b = (b * penalty).norm(p=2)
    #     loss = (- l_a - l_b) * alpha
    #     return loss

    # exp sin
    # def caculate_constarin(self,gamma_m,beta,alpha):
    #     epsilon =  0.05
    #     pi = 3.14159265358979323846
    #     a1 = self.projection_embedding_a.weight - 1.0
    #     a0 = self.projection_embedding_a.weight - 0.0
    #     a = torch.abs(a1 * a0)
    #     penalty = torch.ones_like(a)
    #     penalty[a > gamma_m] =  torch.sin(beta*a[a > gamma_m]-pi/2)-1
    #     l_a = (a * penalty).norm(p=2)
    #     b1 = self.projection_embedding_b.weight - 1.0
    #     b0 = self.projection_embedding_b.weight - 0.0
    #     b = torch.abs(b1 * b0)
    #     penalty = torch.ones_like(b)
    #     penalty[b > gamma_m] =   torch.sin(beta*b[b > gamma_m]-pi/2)-1
    #     l_b = (b * penalty).norm(p=2)
    #     loss = ( - l_a - l_b) * alpha
    #     return loss

    # set
    # def caculate_constarin(self,gamma_m,beta,alpha):
    #     epsilon =  0.05
    #     pi = 3.14159265358979323846

    #     a1 = self.projection_embedding_a.weight - 1.0

    #     a0 = self.projection_embedding_a.weight - 0.0
       

    #     detach = self.projection_embedding_a.weight[a1 > gamma_m].detach()
    #     detach[:] = 1

    #     detach = self.projection_embedding_a.weight[a0 > gamma_m].detach()
    #     detach[:] = 0
      
    #     b1 = self.projection_embedding_b.weight - 1.0
    #     b0 = self.projection_embedding_b.weight - 0.0

    #     detach = self.projection_embedding_b.weight[b1 > gamma_m].detach()
    #     detach[:] = 1

    #     detach = self.projection_embedding_b.weight[b0 > gamma_m].detach()
    #     detach[:] = 0

    #     h1 = self.projection_embedding_a.weight.register_hook(lambda x: modify_grad(x, (a0 < gamma_m) | (a_pi < gamma_m)))
    #     h2 = self.projection_embedding_a.weight.register_hook(lambda x: modify_grad(x, (a0 < gamma_m) | (a_pi < gamma_m)))
    #     return h1,h2
    
    # def caculate_constarin(self,gamma_m,beta,alpha):
    #     epsilon =  0.05
    #     pi = 3.14159265358979323846
    #     a1 = self.projection_embedding_a.weight - 1.0
    #     a0 = self.projection_embedding_a.weight - 0.0
    #     a = torch.abs(a1 * a0)
    #     penalty = torch.ones_like(a)

    #     a = self.projection_embedding_a.weight[a > gamma_m].detach()
    #     a[:] = gamma_m

    #     penalty[a > gamma_m] =  torch.sin(beta*a[a > gamma_m]-pi/2)-1
    #     l_a = (a * penalty).norm(p=2)
    #     b1 = self.projection_embedding_b.weight - 1.0
    #     b0 = self.projection_embedding_b.weight - 0.0
    #     b = torch.abs(b1 * b0)
    #     penalty = torch.ones_like(b)
    #     penalty[b > gamma_m] =   torch.sin(beta*b[b > gamma_m]-pi/2)-1
    #     l_b = (b * penalty).norm(p=2)
    #     loss = ( - l_a - l_b) * alpha
    #     return loss

    # exp-8
    def caculate_constarin_exp8(self,gamma_m,beta,alpha,gamma):
      pi = 3.14159265358979323846
      rot_phase = self.relation_embedding.weight/(self.embedding_range/pi)
      a0 = rot_phase + pi  # phase to -pi
      a0 = torch.abs(a0)
      a_pi = rot_phase - pi    # phase to pi
      a_pi = torch.abs(a_pi)

      penalty = torch.zeros_like(a0)
      penalty[a0 < gamma_m] =  (gamma_m - a0[a0 < gamma_m])  * beta
      l_1 = (penalty).norm(p=1)   
      penalty = torch.zeros_like(a_pi)
      penalty[a_pi < gamma_m] = (gamma_m - a_pi[a_pi< gamma_m]) * beta
      l_2 = (penalty).norm(p=1)

      a_n = torch.norm(rot_phase,p=1,dim=-1)
      penalty = torch.zeros_like(a_n)
      penalty[a_n < gamma] = (gamma - a_n[a_n < gamma]) * beta
      l_3 = (penalty).norm(p=1)
      loss = (-l_1 - l_2 + l_3) * alpha
      return loss

    def caculate_constarin_phare(self,gamma_m,beta,alpha,gamma,leng_min,constype='exp6'):
        if constype == 'exp6':
            return self.caculate_constarin_exp6(gamma_m,beta,alpha,gamma)
        elif constype == 'exp8':
            return self.caculate_constarin_exp8(gamma_m,beta,alpha,gamma)
        elif constype == 'set':
            return self.caculate_constarin_set(gamma_m,beta,alpha,gamma)
        elif constype == 'sin':
            return self.caculate_constarin_sin(gamma_m,beta,alpha,gamma)
        return None
    
    def caculate_constarin_exp6(self,gamma_m,beta,alpha,gamma):
        epsilon =  0.05
        pi = 3.14159265358979323846
        rot_phase = self.relation_embedding.weight/(self.embedding_range/pi)
        a0 = rot_phase + pi  # phase to -pi
        a0 = torch.abs(a0)
        a_pi = rot_phase - pi    # phase to pi
        a_pi = torch.abs(a_pi)

        penalty = torch.zeros_like(a0)
        penalty[a0 < gamma_m] =  1.0 / (a0[a0 < gamma_m]  * beta + epsilon)
        l_1 = (penalty).norm(p=1)   
        penalty = torch.zeros_like(a_pi)
        penalty[a_pi < gamma_m] = 1.0 / (a_pi[a_pi< gamma_m]  * beta + epsilon)
        l_2 = (penalty).norm(p=1)
        a_n = torch.norm(rot_phase,p=1,dim=-1)
        penalty = torch.zeros_like(a_n)
        penalty[a_n < gamma] =1.0 / (a_n[a_n< gamma]  * beta + epsilon)
        l_3 = (penalty).norm(p=1)
        loss = (-l_1 - l_2 + l_3) * alpha
        return loss

    def caculate_constarin_sin(self,gamma_m,beta,alpha,gamma):
        epsilon =  0.05
        pi = 3.14159265358979323846
        rot_phase = self.relation_embedding.weight/(self.embedding_range/pi)
        a0 = rot_phase + pi  # phase to -pi
        a0 = torch.abs(a0)
        a_pi = rot_phase - pi    # phase to pi
        a_pi = torch.abs(a_pi)

        penalty = torch.zeros_like(a0)
        penalty[a0 < gamma_m] =  torch.sin(beta*(a0[a0 < gamma_m])-pi/2)-1
        l_1 = (penalty).norm(p=1)   
        penalty = torch.zeros_like(a_pi)
        penalty[a_pi < gamma_m] = torch.sin(beta*(a_pi[a_pi < gamma_m])-pi/2)-1
        l_2 = (penalty).norm(p=1)
        a_n = torch.norm(rot_phase,p=1,dim=-1)
        penalty = torch.zeros_like(a_n)
        penalty[a_n < gamma] =  torch.sin(beta*(a_n[a_n < gamma_m])-pi/2)-1
        l_3 = (penalty).norm(p=1)
        loss = (-l_1 - l_2 + l_3) * alpha
        return loss

    def caculate_constarin_set(self,gamma_m,beta,alpha,gamma=0.5):
        pi = 3.14159265358979323846
        rot_phase = self.relation_embedding.weight/(self.embedding_range/pi)

        a0 = rot_phase + pi  # phase to -pi
        a0 = torch.abs(a0)

        a_pi = rot_phase - pi    # phase to pi
        a_pi = torch.abs(a_pi)

        detach_a = self.relation_embedding.weight[a0 < gamma_m].detach()
        detach_a[:] = -pi*((self.embedding_range/pi))
        detach_b = self.relation_embedding.weight[a_pi < gamma_m].detach()
        detach_b[:] = pi*((self.embedding_range/pi))

        h = self.relation_embedding.weight.register_hook(lambda x: modify_grad(x, (a0 < gamma_m) | (a_pi < gamma_m)))
        
        a_n = torch.norm(rot_phase,p=1,dim=-1)
        penalty = torch.zeros_like(a_n)
        penalty[a_n < gamma] = (gamma - a_n[a_n < gamma]) * beta
        l_3 = (penalty).norm(p=1)
        loss = (l_3) * alpha
        return loss,h