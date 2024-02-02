

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TransE(nn.Module):
    def __init__(self,n_entity, n_relation, dim, p_norm=1, gamma=None):
        super(TransE,self).__init__()
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.dim = dim
        self.p_norm = p_norm
        self.entity_embedding = nn.Embedding(n_entity,dim)
        self.relation_embedding = nn.Embedding(n_relation,dim)
        
        if gamma != None:
            embedding_range = (gamma + 2)/dim
            nn.init.uniform_(
                tensor=self.entity_embedding.weight,
                a=-embedding_range,
                b=embedding_range
                )
            nn.init.uniform_(
                tensor=self.relation_embedding.weight,
                a=-embedding_range,
                b=embedding_range
                )
        else:
            init_range =  6.0 / math.sqrt(self.dim)
            nn.init.uniform_(self.entity_embedding.weight, -init_range, init_range)
            nn.init.uniform_(self.relation_embedding.weight, -init_range, init_range)
         

    # 这里可以实现从其他格式的文件读取data
    def init_model(self):
        pass

    def save_model_embedding(self):
        return {
            'entity_embedding':self.entity_embedding.weight,
            'relation_embedding':self.relation_embedding.weight
        }
    
    def normalize(self, head, relation, tail):
        head = F.normalize(head, p=2,dim=-1)
        relation = F.normalize(relation, p=2,dim=-1)
        tail = F.normalize(tail, p=2,dim=-1)
        return head,relation,tail

    
    def score_function(self, head, relation, tail):
        # head, relation, tail = self.normalize(head, relation, tail)
        if head.shape[1] == relation.shape[1]:  # hr_t
            score = (head + relation) - tail
        else:
            score = head + (relation - tail)
        score = torch.norm(score, p=self.p_norm, dim=-1)
        return -score

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
            head = self.entity_embedding(h).squeeze()
            tail = self.entity_embedding(t).squeeze()   
            relation = self.relation_embedding(r).squeeze() 
        elif mode =='hr_all':
            head = self.entity_embedding(h).unsqueeze(1)
            tail = self.entity_embedding.weight.unsqueeze(0)
        elif mode =='all_rt':
            head = self.entity_embedding.weight.unsqueeze(0)
            tail = self.entity_embedding(t).unsqueeze(1)
        return self.score_function(head, relation,tail)
    
    def predict(self,h,r,t,mode='hrt'):
        head = self.entity_embedding(h).squeeze()
        tail = self.entity_embedding(t).squeeze()   
        relation = self.relation_embedding(r).squeeze() 
        return self.score_function(head, relation,tail)