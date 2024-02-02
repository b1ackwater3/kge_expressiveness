
import torch
import torch.nn as nn
import torch.nn.functional as F

def modify_grad(x, inds):
    x[inds] = 0 
    return x
class TransH(nn.Module):
    
    def __init__(self,n_entity, n_relation, dim, gamma=None, p_norm=1,dropout=0):
        super(TransH, self).__init__()
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.p_norm = p_norm

        self.entity_embedding = nn.Embedding(n_entity,dim)
        self.relation_embedding = nn.Embedding(n_relation,dim)
        self.relation_norm = nn.Embedding(n_relation,dim)
        print("TransH DropOut %f" % dropout)
        self.dropout = nn.Dropout(dropout)
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
            nn.init.uniform_(
                tensor=self.relation_norm.weight,
                a=-self.embedding_range,
                b=self.embedding_range
                )
        else: 
            nn.init.xavier_uniform_(self.entity_embedding.weight)
            nn.init.xavier_uniform_(self.relation_embedding.weight)
            nn.init.xavier_uniform_(self.relation_norm.weight)

    def score_function(self,head, relation, relation_norm, tail):
        def transfer(entity, norm):
            norm = F.normalize(norm, 2, -1) # (batch, 1, dim)
            if norm.shape[0] != entity.shape[0]:    # to all 
                entity = entity.reshape(-1,entity.shape[-1])
                norm = norm.reshape(-1,norm.shape[-1])
                project = torch.mm(norm,entity.transpose(1,0)).unsqueeze(2)
                norm = norm.unsqueeze(1)
                entity = entity.unsqueeze(0)
                entity = entity - torch.bmm(project,norm)
                return entity            # batch*ne*dim
            else:
                project = torch.bmm(entity,norm.transpose(2,1))
                entity = entity - torch.bmm(project,norm)
                return entity

        head = transfer(head,relation_norm)
        tail = transfer(tail,relation_norm)
        score = head + relation - tail
        score = torch.norm(score, p=self.p_norm, dim=-1)
        return -score


    def forward(self, h, r, t,  mode='hrt'):
        relation = self.relation_embedding(r).unsqueeze(1)
        relation_norm = self.relation_norm(r).unsqueeze(1)
        if mode=='hr_t':
            batch_size  = t.shape[0]
            negative_size = t.shape[1]
            head = self.entity_embedding(h).unsqueeze(1)
            tail = self.entity_embedding(t.reshape(-1,1)).reshape(batch_size,negative_size,-1)
        elif mode == 'h_rt':
            batch_size  = h.shape[0]
            negative_size = h.shape[1]
            head = self.entity_embedding(h.reshape(-1,1)).reshape(batch_size,negative_size,-1)
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
        
        return self.score_function(head, relation,relation_norm,tail)
        

    def caculate_constarin_exp8(self,gamma_m,beta,alpha,gamma=0.5):
      a_n = torch.norm(self.relation_embedding.weight,p=1,dim=-1)
      penalty = torch.zeros_like(a_n)
      penalty[a_n < gamma_m] = (gamma_m - a_n[a_n < gamma_m]) * beta
      l_3 = (penalty).norm(p=1)
      loss = (-l_3) * alpha
      return loss
   
    def caculate_constarin_exp6(self,gamma_m,beta,alpha,gamma=0.5):
        epsilon = 0.05
        a_n = torch.norm(self.relation_embedding.weight,p=1,dim=-1)
        penalty = torch.zeros_like(a_n)
        penalty[a_n < gamma_m] = 1/(a_n[a_n < gamma_m]* beta + epsilon) 
        l_3 = (penalty).norm(p=1)
        loss = (-l_3) * alpha
        return loss

    # setting 0
    def caculate_constarin_set(self,gamma_m,beta,alpha,gamma=0.5):
        a_n = torch.norm(self.relation_embedding.weight,p=1,dim=-1)
        penalty = torch.zeros_like(a_n)
        a_n[a_n < gamma_m] = a_n[a_n < gamma_m].detach()
        a_n[a_n < gamma_m] = 0
        tt = self.relation_embedding.weight.register_hook(lambda x: modify_grad(x, a_n < gamma_m))
        return 0,tt

    def caculate_constarin_sin(self,gamma_m,beta,alpha,gamma=0.5):
        a_n = torch.norm(self.relation_embedding.weight,p=1,dim=-1)
        penalty = torch.zeros_like(a_n)
        PI = 3.1415826
        penalty[a_n < gamma_m] = torch.sin(beta*(a_n[a_n < gamma_m])-PI/2)-1
        l_3 = (penalty).norm(p=1)
        loss = (-l_3) * alpha
        return loss

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