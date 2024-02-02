

import torch
import torch.nn as nn
import torch.nn.functional as F

def modify_grad(x, inds):
    x[inds] = 0 
    return x

class PairRE(nn.Module):
    def __init__(self,n_entity, n_relation, dim, p_norm=1,gamma=None,zero_dim=0):
        super(PairRE,self).__init__()
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.p_norm = p_norm
        self.zero_dim = dim - zero_dim

        self.entity_embedding = nn.Embedding(n_entity,dim)

        # 每个关系再加上一个filter
        self.relation_embedding = nn.Embedding(n_relation,dim*2)
        
        # if gamma != None:
        #     init_range = (gamma + 2.0) / dim
        #     nn.init.uniform_(
        #         tensor=self.entity_embedding.weight,
        #         a=-init_range,
        #         b=init_range
        #     )
        #     nn.init.uniform_(
        #         tensor=self.relation_embedding.weight,
        #         a=-init_range,
        #         b=init_range
        #     )
        # else:
        nn.init.xavier_uniform_(self.entity_embedding.weight)
        nn.init.xavier_uniform_(self.relation_embedding.weight)

    def init_model(self):
        pass

    def save_model_embedding(self):
        embedding_map = {
            'relation_emb': self.relation_embedding.weight, 
            'entity_emb'  : self.entity_embedding.weight                              
        }
        return embedding_map
    

  
    ### 实现exp6
    def caculate_constarin_exp6(self,gamma_m,beta,alpha,gamma=0.5,leng_min=0.01):
        r_h,r_t = torch.chunk(self.relation_embedding.weight,2,dim=-1)

        sub = torch.abs(r_h - r_t) #
        add =  torch.abs(r_h + r_t) #
        
        epsilon = 0.05

        penalty = torch.zeros_like(sub)
        penalty[sub < gamma_m] = 1/(sub[sub < gamma_m]* beta + epsilon)   # 
        l_1 = (penalty).norm(p=1)                                         # sub 越小，L1 值越大, 
        
        penalty = torch.zeros_like(add)
        penalty[add < gamma_m] = 1/(add[add < gamma_m] * beta + epsilon)
        l_2 = (penalty).norm(p=1)
        
        # 对 rh = rt or rh = -rt 的关系，约产生一定0
        sub = torch.sum(sub,dim=-1)
        add = torch.sum(add,dim=-1)

        
        gamma = sub.shape[-1] * 0.0001
        leng_min = sub.shape[-1] /3 * 0.001
        
        index_of_close_rel = (add < gamma) | (sub < gamma)
        
        r_h_len = torch.norm(r_h, p=1, dim=-1)
        r_t_len = torch.norm(r_t, p=1, dim=-1)

        r_h_len =  r_h_len[index_of_close_rel & (r_h_len.detach()>leng_min)]
        r_t_len = r_t_len[index_of_close_rel & (r_t_len.detach()>leng_min)]
        
        r_h_loss = 0
        r_t_loss = 0
        if len(r_h_len)>0:
            r_h_loss = torch.mean(r_h_len)
        if len(r_t_len) > 0:
            r_t_loss = torch.mean(r_t_len)
        
        loss = -l_1 - l_2 + r_h_loss + r_t_loss
        return loss * alpha

    # exp8
    # def caculate_constarin_exp8(self,gamma_m,beta,alpha,gamma=0.5,leng_min=0.01):

    #     r_h,r_t = torch.chunk(self.relation_embedding.weight,2,dim=-1)
    #     sub = torch.abs(r_h - r_t) #
    #     add =  torch.abs(r_h + r_t) #

    #     sub_max = torch.max(sub,dim=-1)
    #     add_max = torch.max(add,dim=-1)

    #     sub_rel = sub[sub_max.values < gamma_m]
    #     add_rel = add[add_max.values < gamma_m]

    #     penalty = torch.zeros_like(sub_rel)
    #     penalty[sub_rel < gamma_m] = (gamma_m - sub_rel[sub_rel < gamma_m]) * beta
    #     l_1 = (penalty).norm(p=1)
        
    #     penalty = torch.zeros_like(add_rel)
    #     penalty[add_rel < gamma_m] = (gamma_m - add_rel[add_rel < gamma_m]) * beta
    #     l_2 = (penalty).norm(p=1)
        
    #     # 对 rh = rt or rh = -rt 的关系，约产生一定0
    #     sub = torch.sum(sub,dim=-1)
    #     add = torch.sum(add,dim=-1)
        
    #     gamma = sub.shape[-1] * 0.00005  # 判定比较接近
    #     leng_min = sub.shape[-1] /3 * 0.00005  # 

    #     index_of_close_rel = (sub_max.values < gamma_m) | (add_max.values < gamma_m)
        
    #     r_h_len = torch.norm(r_h, p=1, dim=-1)
    #     r_t_len = torch.norm(r_t, p=1, dim=-1)
    #     r_h_len =  r_h_len[index_of_close_rel & (r_h_len.detach()>leng_min)]
    #     r_t_len = r_t_len[index_of_close_rel & (r_t_len.detach()>leng_min)]
        
    #     r_h_loss = 0
    #     r_t_loss = 0
    #     if len(r_h_len)>0:
    #         r_h_loss = torch.mean(r_h_len)
    #     if len(r_t_len) > 0:
    #         r_t_loss = torch.mean(r_t_len)
        
    #     loss = -l_1 -l_2 + r_h_loss + r_t_loss
    #     return loss*alpha

    def caculate_constarin_exp8(self,gamma_m,beta,alpha,gamma=0.5,leng_min=0.01):

        r_h,r_t = torch.chunk(self.relation_embedding.weight,2,dim=-1)
        sub = torch.abs(r_h - r_t) #
        add =  torch.abs(r_h + r_t) #

        penalty = torch.zeros_like(sub)
        penalty[sub < gamma_m] = (gamma_m - sub[sub < gamma_m]) * beta
        l_1 = (penalty).norm(p=1)
        
        penalty = torch.zeros_like(add)
        penalty[add < gamma_m] = (gamma_m - add[add < gamma_m]) * beta
        l_2 = (penalty).norm(p=1)
        # 对 rh = rt or rh = -rt 的关系，约产生一定0
        sub = torch.sum(sub,dim=-1)
        add = torch.sum(add,dim=-1)
        
        gamma = sub.shape[-1] * 0.000005  # 判定比较接近
        leng_min = sub.shape[-1] /3 * 0.000005  # 

        index_of_close_rel = (add < gamma) | (sub < gamma)
        
        r_h_len = torch.norm(r_h, p=1, dim=-1)
        r_t_len = torch.norm(r_t, p=1, dim=-1)
        r_h_len =  r_h_len[index_of_close_rel & (r_h_len.detach()>leng_min)]
        r_t_len = r_t_len[index_of_close_rel & (r_t_len.detach()>leng_min)]
        
        r_h_loss = 0
        r_t_loss = 0
        if len(r_h_len)>0:
            r_h_loss = torch.mean(r_h_len)
        if len(r_t_len) > 0:
            r_t_loss = torch.mean(r_t_len)
        
        loss = -l_1 -l_2  #+ r_h_loss + r_t_loss
        return loss*alpha
    # # 直方图
    def caculate_constarin_set(self,gamma_m,beta,alpha,gamma=0.5,leng_min=0.01):
        r_h,r_t = torch.chunk(self.relation_embedding.weight,2,dim=-1)
        sub = torch.abs(r_h - r_t) #
        add =  torch.abs(r_h + r_t) #
        epsilon = 0.05

        r_h_d = r_h.detach()
        r_t_d = r_t.detach()
        mean1 = (r_t_d[sub < gamma_m] + r_h_d[sub < gamma_m])/2
        mean2 = (r_h_d[add < gamma_m]  -r_t_d[add < gamma_m] )/2

        r_h_d[sub < gamma_m] = mean1
        r_t_d[sub < gamma_m] = mean1
        r_h_d[add < gamma_m] = mean2
        r_t_d[add < gamma_m] = mean2

        h = r_h.register_hook(lambda x: modify_grad(x, sub < gamma_m | add < gamma_m))
        t = r_t.register_hook(lambda x: modify_grad(x, sub < gamma_m | add < gamma_m))

        # 对 rh = rt or rh = -rt 的关系，约产生一定0
        gamma = sub.shape[-1] * 0.001  # 判定比较接近
        leng_min = sub.shape[-1] /3 * 0.001  # 

        sub = torch.sum(sub,dim=-1)
        add = torch.sum(add,dim=-1)
        index_of_close_rel = (add < gamma) | (sub < gamma)
        r_h_len = torch.norm(r_h, p=1, dim=-1)
        r_t_len = torch.norm(r_t, p=1, dim=-1)
        r_h_len =  r_h_len[index_of_close_rel & (r_h_len.detach()>leng_min)]
        r_t_len = r_t_len[index_of_close_rel & (r_t_len.detach()>leng_min)]
        
        r_h_loss = 0
        r_t_loss = 0
        if len(r_h_len) > 0:
            r_h_loss = torch.mean(r_h_len)
        if len(r_t_len) > 0:
            r_t_loss = torch.mean(r_t_len)
        
        loss = r_h_loss + r_t_loss
        return loss*alpha,h,t
    # sin
    def caculate_constarin_sin(self,gamma_m,beta,alpha,gamma=0.5,leng_min=0.01):
        # print(gamma_m,beta,alpha,gamma,leng_min)  
        PI = 3.14159265358979323846
        r_h,r_t = torch.chunk(self.relation_embedding.weight,2,dim=-1)
        sub = torch.abs(r_h - r_t) #
        add =  torch.abs(r_h + r_t) #

        penalty = torch.zeros_like(sub)
        penalty[sub < gamma_m] = torch.sin(beta*(sub[sub < gamma_m])-PI/2)-1
        l_1 = (penalty).norm(p=1)
        
        penalty = torch.zeros_like(add)
        penalty[add < gamma_m] =  torch.sin(beta*( add[add < gamma_m])-PI/2)-1
        l_2 = (penalty).norm(p=1)
        
        # 对 rh = rt or rh = -rt 的关系，约产生一定0
        gamma = sub.shape[-1] * 0.0001
        leng_min = sub.shape[-1] /3 * 0.001

        sub = torch.sum(sub,dim=-1)
        add = torch.sum(add,dim=-1)
        index_of_close_rel = (add < gamma) | (sub < gamma)
        r_h_len = torch.norm(r_h, p=1, dim=-1)
        r_t_len = torch.norm(r_t, p=1, dim=-1)
        r_h_len =  r_h_len[index_of_close_rel & (r_h_len.detach()>leng_min)]
        r_t_len = r_t_len[index_of_close_rel & (r_t_len.detach()>leng_min)]
        
        r_h_loss = 0
        r_t_loss = 0
        if len(r_h_len)>0:
            r_h_loss = torch.mean(r_h_len)
        if len(r_t_len) > 0:
            r_t_loss = torch.mean(r_t_len)
        loss = -l_1 - l_2 + r_h_loss + r_t_loss
        return loss*alpha

    def caculate_constarin(self,gamma_m,beta,alpha,gamma,leng_min,constype='exp6'):
        if constype == 'exp6':
            return self.caculate_constarin_exp6(gamma_m,beta,alpha,gamma,leng_min)
        elif constype == 'exp8':
            return self.caculate_constarin_exp8(gamma_m,beta,alpha,gamma,leng_min)
        elif constype == 'set':
            return self.caculate_constarin_set(gamma_m,beta,alpha,gamma,leng_min)
        elif constype == 'sin':
            return self.caculate_constarin_sin(gamma_m,beta,alpha,gamma,leng_min)
        return None
    # def caculate_constarin(self):
    #     re_head, re_tail, filter = torch.chunk(self.relation_embedding.weight, 3, dim=-1)
    #     norm_value = torch.norm(filter,dim=-1) -  self.zero_dim

    #     # re_head = re_head[...,:300]
    #     # re_tail = re_tail[...,:300]
    #     # a = torch.norm(re_head,p=2, dim=-1)
    #     # b = torch.norm(re_tail,p=2, dim=-1)
    #     # return torch.sum(a) + torch.sum(b)
    #     return torch.norm(norm_value)

    # def caculate_constarin_sys(self,isMarry,isConnect):
    #     re_head, re_tail = torch.chunk(self.relation_embedding.weight, 2, dim=-1)
    #     # marry = self.entity_embedding(isMarry)
    #     # connect = self.entity_embedding(isConnect)
    #     # # marry = marry**2
    #     # # connect = connect**2
    #     # # marry = torch.sum(torch.var(marry,dim=-1))
    #     # # connect = torch.sum(torch.var(connect,dim=-1))

    #     # # return marry + connect
    #     # relation_loss = torch.sum(torch.norm(re_head**2-re_tail**2))
    #     # return torch.sum(torch.norm(marry-connect)) + relation_loss


    def score_function(self, head, relation, tail):
        re_head, re_tail = torch.chunk(relation, 2, dim=-1)

        head = F.normalize(head, 2, -1)
        tail = F.normalize(tail, 2, -1)

        score = head * re_head - tail * re_tail
        score = - torch.norm(score, p=1, dim=2)
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