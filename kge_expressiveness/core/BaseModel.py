#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseModel(nn.Module):

    def __init__(self, model_name, nentity, nrelation, e_dim, r_dim,  gamma):
        super(BaseModel, self).__init__()

        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.entity_dim = e_dim
        self.relation_dim = r_dim
        self.epsilon = 2.0

        if gamma != None:
            self.gamma = nn.Parameter(
                torch.Tensor([gamma]), 
                requires_grad=False
            )
            self.gamma_flag = True
        else:
            self.gamma_flag = False
        
        self.e_embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / e_dim]), 
            requires_grad=False
        )
        self.r_embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / r_dim]), 
            requires_grad=False
        )

        self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
        self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))

        if self.gamma_flag:
            nn.init.uniform_(
                tensor=self.entity_embedding, 
                a=-self.e_embedding_range.item(), 
                b=self.e_embedding_range.item()
            )
            nn.init.uniform_(
                    tensor=self.relation_embedding, 
                    a=-self.r_embedding_range.item(), 
                    b=self.r_embedding_range.item()
            )
        else:
            nn.init.xavier_uniform_(self.entity_embedding)
            nn.init.xavier_uniform_(self.relation_embedding)
        self.entity_embedding_list = [self.entity_embedding]
        self.relation_embedding_list = [self.entity_embedding]
        self.e_num  = len(self.entity_embedding_list)
        self.r_num = len(self.relation_embedding_list)


    def get_entitys(self, sample, base_index, mode='sigle', batch_size=None,negative_sample_size=None):
        heads = []
        e_index = None
        if mode == 'head-batch' or mode == 'tail-batch':
            e_index = sample.contiguous().reshape(-1)
        else:
            e_index = sample[:,base_index]
        for i in range(self.e_num):
            head = torch.index_select(
                self.entity_embedding_list[i], 
                dim=0, 
                index=e_index
            ).unsqueeze(1)
            if mode == 'head-batch':
                head = head.reshape(batch_size,negative_sample_size,-1)
            heads.append(head)
        return heads
    
    def get_relations(self, sample):
        relations = []
        for i in range(self.r_num):
            relation = torch.index_select(
                self.relation_embedding, 
                dim=0, 
                index=sample[:,1]
            ).unsqueeze(1)
            relations.append(relation)
        return relations

    def handle_sinle(self, sample):
        relations = self.get_relations(sample)
        heads = self.get_entitys(sample,0)
        tails = self.get_entitys(sample,2)
        return heads, relations, tails

    def handle_head_batch(self, sample):
        tail_part, head_part = sample
        relations = []
        batch_size, negative_sample_size = head_part.size(0), head_part.size(1)
        
        heads = self.get_entitys(head_part,0,'head-batch',batch_size,negative_sample_size)
        relations = self.get_relations(tail_part)
        tails = self.get_entitys(tail_part,2)
        return heads, relations, tails

    def handle_tail_batch(self, sample):
        head_part, tail_part = sample
        batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

        heads = self.get_entitys(head_part,0)
        relations = self.get_relations(head_part)
        tails = self.get_entitys(tail_part,2,'tail-batch',batch_size,negative_sample_size)
        return heads, relations, tails

    def handle_1to1(self, sample):
        head = torch.index_select(
            self.entity_embedding, 
            dim=0, 
            index=sample[:, 0]
        ).unsqueeze(1)
        
        relation = torch.index_select(
            self.relation_embedding,
            dim=0,
            index=sample[:, 1]
        ).unsqueeze(1)

        relations = [relation]
        tail = torch.index_select(
            self.entity_embedding,
            dim=0,
            index=sample[:, 2]
        ).unsqueeze(1)
        result,factor =self.ComplEx(head, relations, tail,'1to1')
        return result, factor

    def handle_1ton(self, sample):
        head = torch.index_select(
            self.entity_embedding, 
            dim=0, 
            index=sample[:, 0]
        ).unsqueeze(1)
        
        relation = torch.index_select(
            self.relation_embedding,
            dim=0,
            index=sample[:, 1]
        ).unsqueeze(1)

        relations = [relation]
        tail = self.entity_embedding
        result,factor =self.TuckER(head, relations, tail,'1toN')
        score = torch.sigmoid(result)
        return score,factor

    def forward(self, sample, mode='single'):
        '''
        Forward function that calculate the score of a batch of triples.
        In the 'single' mode, sample is a batch of triple.
        In the 'head-batch' or 'tail-batch' mode, sample consists two part.
        The first part is usually the positive sample.
        And the second part is the entities in the negative samples.
        Because negative samples and positive samples usually share two elements 
        in their triple ((head, relation) or (relation, tail)).
        '''
        if mode == 'single':
            head, relations, tail = self.handle_sinle(sample)
        elif mode == 'head-batch':
            head, relations, tail = self.handle_head_batch(sample)
        elif mode == 'tail-batch':
            head, relations, tail = self.handle_tail_batch(sample)
        elif mode =='1toN':
            head, relations, tail = self.handle_1ton(sample)
        elif mode == '1to1':
            head, relations, tail = self.handle_1to1(sample)
        else:
            raise ValueError('mode %s not supported' % mode)
        score,factor = self.score(head, relations, tail, mode)

        return score,factor

    def score(self, head, relations, tail, mode):
        pass

    def RotPro(self, head, relations, tail,mode):
        relation = relations[0]
        proj_a = relations[1]
        proj_b = relations[2]
        proj_p = relations[3]
        pi = 3.14159265358979323846

        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)  

        # Make phases of relations uniformly distributed in [-pi, pi]

        phase_relation = (relation / (self.embedding_range.item() / pi) )  * self.train_pr_prop

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

        if mode == 'head-batch':
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
        score = self.gamma.item() - score.sum(dim=2)
        return score, None

    def TransE(self, head, relation, tail, mode):
        relation = relation[0]
        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score, None

    def _calc(self, h, t, r, mode):
        if self.gamma_flag:
            h = F.normalize(h, 2, -1)
            r = F.normalize(r, 2, -1)
            t = F.normalize(t, 2, -1)
        if mode == 'head_batch':
            score = h + (r - t)
        else:
            score = (h + r) - t
        score = torch.norm(score, 1, dim=2)
        return score

    def _transfer(self, e, norm):
        norm = F.normalize(norm, p = 2, dim = -1)
        if e.shape[0] != norm.shape[0]:
            e = e.reshape(-1, norm.shape[0], e.shape[-1])
            norm = norm.reshape(-1, norm.shape[0], norm.shape[-1])
            e = e - torch.sum(e * norm, -1, True) * norm
            return e.reshape(-1, e.shape[-1])
        else:
            return e - torch.sum(e * norm, -1, True) * norm

    def TransH(self, head, relation, tail, mode):
        r = relation[0]
        r_norm = relation[1]
        h = self._transfer(head, r_norm)
        t = self._transfer(tail, r_norm)
        score = self._calc(h ,t, r, mode)
        if self.gamma_flag:
            return self.gamma.item() - score, None
        else:
            return score, None
    
    def _transfer_r(self, e, r_transfer):
        return torch.bmm(e, r_transfer)

    def TransR(self, head, relation, tail, mode):
        r = relation[0]
        transfer = relation[1].reshape(-1, self.entity_dim, self.relation_dim)
        h = self._transfer_r(head, transfer)
        t = self._transfer_r(tail, transfer)
        score = self._calc(h ,t, r, mode)
        if self.gamma_flag:
            return self.gamma.item() - score, None
        else:
            return score, None

    def HoLE(self, head, relation, tail, mode):
        relation = relation[0]
        def ccorr(a, b):
            return  torch.fft.ifft(torch.conj(torch.fft.fft(a))*torch.fft.fft(b)).real 
        score = torch.sum(ccorr(head,tail)*relation, -1)
        return score, None

    def PairRE(self, head, relation, tail, mode):
        relation=relation[0]
        re_head, re_tail = torch.chunk(relation, 2, dim=2)
        head = F.normalize(head, 2, -1)
        tail = F.normalize(tail, 2, -1)
        score = head * re_head - tail * re_tail
        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score, None

    def DistMult(self, head, relation, tail, mode):
        relation = relation[0]
        if mode == 'head-batch':
            score = head * (relation * tail)
        else:
            score = (head * relation) * tail

        score = score.sum(dim = 2)
        return score, None

    def ComplEx(self, head, relation, tail, mode):
        relation = relation[0]
        tail_copy = tail
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        re_sample_tail, im_sample_tail = torch.tensor(0), torch.tensor(0)
        if mode =='1to1':
            tail = self.entity_embedding
            re_tail, im_tail = torch.chunk(tail,2, dim=1)
            re_sample_tail,im_sample_tail = torch.chunk(tail_copy, 2, dim=2)
        else:
            re_tail, im_tail = torch.chunk(tail, 2, dim=2)
        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            score = re_head * re_score + im_head * im_score
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            if mode == '1to1':
                score = re_score @ re_tail.transpose(0,1) + im_score @ im_tail.transpose(0,1)
            else:
                score = re_score * re_tail + im_score * im_tail
        if mode != '1to1':
            score = score.sum(dim = 2)
        else:
            score = score.squeeze()
        factor = (
            torch.sqrt(re_head**2 + im_head**2),
            torch.sqrt(re_relation**2 + im_relation**2),
            torch.sqrt(re_sample_tail**2 + im_sample_tail**2)
        )
        return score,factor

    def RotatE(self, head, relation, tail, mode):
        relation = relation[0]
        pi = 3.14159265358979323846
        
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        #Make phases of relations uniformly distributed in [-pi, pi]
        phase_relation = relation/(self.embedding_range.item()/pi)
        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)
        if mode == 'head-batch':
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
        score = self.gamma.item() - score.sum(dim = 2)
        return score, None

    def pRotatE(self, head, relation, tail, mode):
        relation = relation[0]
        pi = 3.14159262358979323846
        #Make phases of entities and relations uniformly distributed in [-pi, pi]
        phase_head = head/(self.embedding_range.item()/pi)
        phase_relation = relation/(self.embedding_range.item()/pi)
        phase_tail = tail/(self.embedding_range.item()/pi)

        if mode == 'head-batch':
            score = phase_head + (phase_relation - phase_tail)
        else:
            score = (phase_head + phase_relation) - phase_tail
        score = torch.sin(score)            
        score = torch.abs(score)
        score = self.gamma.item() - score.sum(dim = 2) * self.modulus
        return score, None
    
    def TuckER(self, head, relation, tail, mode):
        relation = relation[0]
        batch_size = len(relation)
        if mode == 'head-batch':
            head = head.reshape(-1, self.entity_dim)
            x = self.bn0(head)
            x = self.input_dropout(x)
            x = x.reshape(batch_size, -1, self.entity_dim)
            relation = relation.reshape(-1, self.relation_dim)
            W_mat = torch.mm(relation, self.W.reshape(self.relation_dim, -1))
            W_mat = W_mat.reshape(-1, self.entity_dim, self.entity_dim)
            x = torch.bmm(x, W_mat)
            x = x.reshape(-1, self.entity_dim)      
            x = self.bn1(x)
            x = self.hidden_dropout2(x)
            x = x.reshape(batch_size, -1, self.entity_dim)
            x = torch.bmm(x, tail.permute(0,2,1))
            return torch.squeeze(x,dim=2), None
        else:
            head = head.reshape(-1, self.entity_dim)
            relation = relation.reshape(-1, self.relation_dim)
            x = self.bn0(head)
            x = self.input_dropout(x)
            # 转为三维的张量
            x = x.reshape(-1, 1, self.entity_dim)
            W_mat = torch.mm(relation, self.W.reshape(self.relation_dim, -1))
            W_mat = W_mat.reshape(-1, self.entity_dim,self.entity_dim)
            W_mat = self.hidden_dropout1(W_mat)
            x = torch.bmm(x, W_mat) 
            x = x.reshape(-1, self.entity_dim)      
            x = self.bn1(x)
            x = self.hidden_dropout2(x)
            # 到这里 恢复之前的shape
            if mode == '1toN':
                return torch.mm(x, tail.permute(1,0)), None
            x = x.reshape(-1, 1, self.entity_dim)
            x = torch.bmm(x, tail.permute(0,2,1))
            return torch.squeeze(x,dim=1), None
