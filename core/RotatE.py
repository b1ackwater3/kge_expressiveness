

import torch
import torch.nn as nn

def modify_grad(x, inds):
    x[inds] = 0 
    return x

class RotatE(nn.Module):
    def __init__(self, n_entity, n_relation, dim, gamma=None,p_norm=1):
        super(RotatE,self).__init__()

        self.n_entity = n_entity
        self.n_relation = n_relation
        self.epsilon = 2
        self.entity_dim = dim*2
        self.relation_dim = dim
        self.entity_embedding = nn.Embedding(n_entity, self.entity_dim)
        self.relation_embedding = nn.Embedding(n_relation,self.relation_dim)

        if gamma != None:
            self.embedding_range = (gamma + 2)/dim
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
            nn.init.xavier_uniform_(self.entity_embedding.weight)
            nn.init.xavier_uniform_(self.relation_embedding.weight)
       
    def init_model(self):
        pass
    
    def score_function(self, head, relation, tail):
        pi = 3.14159265358979323846
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        # CreateMake phases of relations uniformly distributed in [-pi, pi]
        phase_relation = relation/(self.embedding_range/pi)
        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if  tail.shape[1] == relation.shape[1]:
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail
        score = torch.stack([re_score, im_score], dim = 0)
        score = score.norm(dim = 0)
        score =  - score.sum(dim = 2)
        return score

    def forward(self, h,r,t, mode='hrt'):
        head = None
        tail = None
        relation = self.relation_embedding(r).unsqueeze(1)
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
        elif mode =='hr_all':
            head = self.entity_embedding(h).unsqueeze(1)
            tail = self.entity_embedding.weight.unsqueeze(0)
        elif mode =='all_rt':
            head = self.entity_embedding.weight.unsqueeze(0)
            tail = self.entity_embedding(t).unsqueeze(1)
        return self.score_function(head, relation,tail)

    # 不要靠近0，靠近+-pi。 不要干扰其他的
    # def caculate_constarin(self,gamma_m,beta,alpha):
    #     pi = 3.14159265358979323846
    #     normal_emb = self.relation_embedding.weight/(self.embedding_range/pi)
    #     a0 = normal_emb + pi
    #     a0 = torch.abs(a0)
    #     a_pi = normal_emb - pi
    #     a_pi = torch.abs(a_pi)

    #     penalty = torch.zeros_like(a0)
    #     penalty[a0 < gamma_m] = beta
    #     l_1 = (a0 * penalty).norm(p=2)
        
    #     penalty = torch.zeros_like(a_pi)
    #     penalty[a_pi < gamma_m] = beta
    #     l_2 = (a_pi * penalty).norm(p=2)

    #     penalty = torch.zeros_like(normal_emb)
    #     an = torch.abs(normal_emb)
    #     penalty[an < gamma_m] = beta
    #     l_3 = (an * penalty).norm(p=2)
        
    #     loss = (l_1 + l_2 - l_3) * alpha
    #     return loss

    # def caculate_constarin(self,gamma_m,beta,alpha):
    #     pi = 3.14159265358979323846
    #     rot_phase = self.relation_embedding.weight/(self.embedding_range/pi)
    #     a0 = rot_phase + pi
    #     a0 = torch.abs(a0)
    #     a_pi = rot_phase - pi
    #     a_pi = torch.abs(a_pi)

    #     penalty = torch.zeros_like(a0)
    #     penalty[a0 < gamma_m] =  - a0[a0 < gamma_m]  * beta
    #     l_1 = (penalty).norm(p=1)   

    #     penalty = torch.zeros_like(a_pi)
    #     penalty[a_pi < gamma_m] = - a_pi[a_pi < gamma_m] * beta
    #     l_2 = (penalty).norm(p=1)
        
    #     penalty = torch.zeros_like(rot_phase)
    #     a_n = torch.abs(rot_phase)
    #     penalty[a_n < gamma_m] = a_n[a_n < gamma_m] * beta
    #     l_3 = (penalty).norm(p=1)

    #     loss = (l_1 + l_2 - l_3) * alpha
    #     return loss

    # exp 5
    # def caculate_constarin(self,gamma_m,beta,alpha):
    #     pi = 3.14159265358979323846
    #     rot_phase = self.relation_embedding.weight/(self.embedding_range/pi)
    #     a0 = rot_phase + pi  # phase to -pi
    #     a0 = torch.abs(a0)
    #     a_pi = rot_phase - pi    # phase to pi
    #     a_pi = torch.abs(a_pi)
    #     epsilon =  0.05
    #     penalty = torch.zeros_like(a0)
    #     penalty[a0 < gamma_m] =  1.0 / (a0[a0 < gamma_m]  * beta + epsilon )
    #     l_1 = (penalty).norm(p=1)   
    #     penalty = torch.zeros_like(a_pi)
    #     penalty[a_pi < gamma_m] = 1.0 / (a_pi[a_pi < gamma_m]  * beta + epsilon )
    #     l_2 = (penalty).norm(p=1)
    #     loss = (-l_1 -l_2) * alpha
    #     return loss
    # exp6 
    # def caculate_constarin(self,gamma_m,beta,alpha):
    #     pi = 3.14159265358979323846
    #     rot_phase = self.relation_embedding.weight/(self.embedding_range/pi)
    #     a0 = rot_phase + pi  # phase to -pi
    #     a0 = torch.abs(a0)
    #     a_pi = rot_phase - pi    # phase to pi
    #     a_pi = torch.abs(a_pi)
    #     epsilon =  0.05
    #     penalty = torch.zeros_like(a0)
    #     penalty[a0 < gamma_m] =  1.0 / (a0[a0 < gamma_m]  * beta + epsilon)
    #     l_1 = (penalty).norm(p=1)   
    #     penalty = torch.zeros_like(a_pi)
    #     penalty[a_pi < gamma_m] = 1.0 / (a_pi[a_pi < gamma_m]  * beta + epsilon)
    #     l_2 = (penalty).norm(p=1)
        
    #     penalty = torch.zeros_like(rot_phase)
    #     a_n = torch.abs(rot_phase)
    #     penalty[a_n < gamma_m] = 1.0 / (a_n[a_n < gamma_m] * beta + epsilon)
    #     l_3 = (penalty).norm(p=1)
    #     loss = (-l_1 - l_2 + l_3) * alpha
    #     return loss

    # exp4
    # def caculate_constarin(self,gamma_m,beta,alpha):
    #     pi = 3.14159265358979323846
    #     normal_emb = self.relation_embedding.weight/(self.embedding_range/pi)
    #     a0 = normal_emb + pi
    #     a0 = torch.abs(a0)
    #     a_pi = normal_emb - pi
    #     a_pi = torch.abs(a_pi)
    
    #     penalty = torch.zeros_like(a0)
    #     penalty[a0 < gamma_m] = beta
    #     l_1 = (a0 * penalty).norm(p=2)
            
    #     penalty = torch.zeros_like(a_pi)
    #     penalty[a_pi < gamma_m] = beta
    #     l_2 = (a_pi * penalty).norm(p=2)

    #     penalty = torch.zeros_like(normal_emb)
    #     an = torch.abs(normal_emb)
    #     penalty[an < gamma_m] = beta
    #     l_3 = (an * penalty).norm(p=2)
            
    #     loss = (l_1 + l_2 - l_3) * alpha
    #     return loss
    
    # exp7
    # def caculate_constarin(self,gamma_m,beta,alpha):
    #     pi = 3.14159265358979323846
    #     rot_phase = self.relation_embedding.weight/(self.embedding_range/pi)
    #     a0 = rot_phase + pi  # phase to -pi
    #     a0 = torch.abs(a0)
    #     a_pi = rot_phase - pi    # phase to pi
    #     a_pi = torch.abs(a_pi)

    #     penalty = torch.zeros_like(a0)
    #     penalty[a0 < gamma_m] =  (gamma_m - a0[a0 < gamma_m])  * beta
    #     l_1 = (penalty).norm(p=1)   
    #     penalty = torch.zeros_like(a_pi)
    #     penalty[a_pi < gamma_m] = (gamma_m - a_pi[a_pi< gamma_m]) * beta
    #     l_2 = (penalty).norm(p=1)
        
    #     penalty = torch.zeros_like(rot_phase)
    #     a_n = torch.abs(rot_phase)
    #     penalty[a_n < gamma_m] = (gamma_m - a_n[a_n < gamma_m]) * beta
    #     l_3 = (penalty).norm(p=1)
    #     loss = (-l_1 - l_2 + l_3) * alpha
    #     return loss
    def caculate_constarin(self,gamma_m,beta,alpha,gamma,leng_min,constype='exp6'):
        if constype == 'exp6':
            return self.caculate_constarin_exp6(gamma_m,beta,alpha,gamma)
        elif constype == 'exp8':
            return self.caculate_constarin_exp8(gamma_m,beta,alpha,gamma)
        elif constype == 'set':
            return self.caculate_constarin_set(gamma_m,beta,alpha,gamma)
        elif constype == 'sin':
            return self.caculate_constarin_sin(gamma_m,beta,alpha,gamma)
        return None
 
    
    # set loss function
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
    

    # exp 6 : exp6 + other
    def caculate_constarin_exp6(self,gamma_m,beta,alpha,gamma=0.8):
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
        lo = (-l_1 - l_2).item()
        lz = l_3.item()
        log = {
            "lo": lo,
            "lz": lz
        }
        return loss,log

    # exp-8
    def caculate_constarin_exp8(self,gamma_m,beta,alpha,gamma=0.5):
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

        lo = (-l_1 - l_2).item()
        lz = l_3.item()
        log = {
            "lo": lo,
            "lz": lz
        }
        loss = (-l_1 - l_2 + l_3) * alpha
        return loss,log

    # exp sin
    def caculate_constarin_sin(self,gamma_m,beta,alpha,gamma=0.8):
        epsilon =  0.05
        pi = 3.14159265358979323846
        PI = 3.14159265358979323846

        rot_phase = self.relation_embedding.weight/(self.embedding_range/pi)
        a0 = rot_phase + pi  # phase to -pi
        a0 = torch.abs(a0)
        a_pi = rot_phase - pi    # phase to pi
        a_pi = torch.abs(a_pi)
        penalty = torch.zeros_like(a0)
        penalty[a0 < gamma_m] =  torch.sin(beta*(a0[a0 < gamma_m])-PI/2)-1
        l_1 = (penalty).norm(p=1) 

        penalty = torch.zeros_like(a_pi)
        penalty[a_pi < gamma_m] = torch.sin(beta*(a_pi[a_pi < gamma_m])-PI/2)-1
        l_2 = (penalty).norm(p=1)

        a_n = torch.norm(rot_phase,p=1,dim=-1)
        penalty = torch.zeros_like(a_n)
        penalty[a_n < gamma] = 1.0 / (a_n[a_n< gamma]  * beta + epsilon)
        l_3 = (penalty).norm(p=1)
        loss = (-l_1 - l_2 + l_3) * alpha

        lo = (-l_1 - l_2).item()
        lz = l_3.item()
        log = {
            "lo": lo,
            "lz": lz
        }
        return loss,log
    
    def caculate_constarin_sub(self,gamma_m,beta,alpha,gamma,rel_index,constype='exp6',):
        if constype == 'exp6':
            return self.caculate_constarin_exp6_sub(gamma_m,beta,alpha,gamma,rel_index)
        elif constype == 'exp8':
            return self.caculate_constarin_exp8_sub(gamma_m,beta,alpha,gamma,rel_index)
        elif constype == 'set':
            return self.caculate_constarin_set_sub(gamma_m,beta,alpha,gamma,rel_index)
        elif constype == 'sin':
            return self.caculate_constarin_sin_sub(gamma_m,beta,alpha,gamma,rel_index)
        return None
 
    
    # set loss function
    def caculate_constarin_set_sub(self,gamma_m,beta,alpha,gamma=0.5,rel_index=None):
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
    

    # exp 6 : exp6 + other
    def caculate_constarin_exp6_sub(self,gamma_m,beta,alpha,gamma=0.8,rel_index=None):
        epsilon =  0.05
        pi = 3.14159265358979323846
        rot_phase = self.relation_embedding(rel_index)/(self.embedding_range/pi)
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
        lo = (-l_1 - l_2).item()
        lz = l_3.item()
        log = {
            "lo": lo,
            "lz": lz
        }
        return loss,log

    # exp-8
    def caculate_constarin_exp8_sub(self,gamma_m,beta,alpha,gamma=0.5,rel_index=None):
        pi = 3.14159265358979323846
        rot_phase =self.relation_embedding(rel_index)/(self.embedding_range/pi)
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

        lo = (-l_1 - l_2).item()
        lz = l_3.item()
        log = {
            "lo": lo,
            "lz": lz
        }
        loss = (-l_1 - l_2 + l_3) * alpha
        return loss,log

    # exp sin
    def caculate_constarin_sin_sub(self,gamma_m,beta,alpha,gamma=0.8,rel_index=None):
        epsilon =  0.05
        pi = 3.14159265358979323846
        PI = 3.14159265358979323846

        rot_phase =self.relation_embedding(rel_index)/(self.embedding_range/pi)
        a0 = rot_phase + pi  # phase to -pi
        a0 = torch.abs(a0)
        a_pi = rot_phase - pi    # phase to pi
        a_pi = torch.abs(a_pi)
        penalty = torch.zeros_like(a0)
        penalty[a0 < gamma_m] =  torch.sin(beta*(a0[a0 < gamma_m])-PI/2)-1
        l_1 = (penalty).norm(p=1) 

        penalty = torch.zeros_like(a_pi)
        penalty[a_pi < gamma_m] = torch.sin(beta*(a_pi[a_pi < gamma_m])-PI/2)-1
        l_2 = (penalty).norm(p=1)

        a_n = torch.norm(rot_phase,p=1,dim=-1)
        penalty = torch.zeros_like(a_n)
        penalty[a_n < gamma] = 1.0 / (a_n[a_n< gamma]  * beta + epsilon)
        l_3 = (penalty).norm(p=1)
        loss = (-l_1 - l_2 + l_3) * alpha

        lo = (-l_1 - l_2).item()
        lz = l_3.item()
        log = {
            "lo": lo,
            "lz": lz
        }
        return loss,log