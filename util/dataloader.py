#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from cmd import IDENTCHARS
from concurrent.futures import thread

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os

class BoxTripleDataset(Dataset):
    def __init__(self, triples, nentity, nrelation,n_size,n_value=1, mode='hr_t',random=True):
        self.nentity = nentity
        self.nrelation = nrelation
        self.triples = triples
        self.mode = mode
        self.n_size = n_size
        self.build_dataset()
        lable =  torch.zeros(n_size+1) 
        nn.init.constant_(lable, n_value)
        lable[0] = 1
        self.lable = lable
        self.random = random

    def build_dataset(self):
        true_dict = {}
        for h,r,t in self.triples:
            true_dict[h] = t
            true_dict[t] = h
        self.true_dic = true_dict
        
    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        head,relation,tail =self.triples[idx]
        sample_r = torch.LongTensor([relation]).expand(self.n_size+1)
        if self.mode == 'hr_t':
            sample_h = torch.LongTensor([head]).expand(self.n_size+1)
            n_list = [[tail]]
        else:
            sample_t = torch.LongTensor([tail]).expand(self.n_size+1)
            n_list = [[head]]

        n_size = 0
        replace_index = 0
        while n_size < self.n_size:
            rdm_words = np.random.randint(0, self.nentity, self.n_size*2)
            if self.mode == 'hr_t':
                mask = np.in1d(
                    rdm_words, 
                    self.true_dic[head], 
                    assume_unique=True, 
                    invert=True
                )
            else: # 如果小于则替换头实体
                mask = np.in1d(
                    rdm_words, 
                    self.true_dic[tail], 
                    assume_unique=True, 
                    invert=True
                )
            rdm_words = rdm_words[mask] # filter true triples
            n_list.append(rdm_words)
            n_size += rdm_words.size
        
        negative_sample = np.concatenate(n_list)[:self.n_size+1]

        if self.mode == 'hr_t':
            sample_t =  torch.LongTensor(negative_sample)
        else:
            sample_h = torch.LongTensor(negative_sample)

        if self.random:
            shuffle_idx = torch.randperm(sample_h.nelement())
            return sample_h[shuffle_idx],sample_r[shuffle_idx], sample_t[shuffle_idx], self.lable[shuffle_idx],self.mode
        else:
            return sample_h ,sample_r, sample_t, self.lable,self.mode
      
    @staticmethod
    def collate_fn(data):
        h = torch.stack([_[0] for _ in data], dim=0)
        r = torch.stack([_[1] for _ in data], dim=0)
        t = torch.stack([_[2] for _ in data], dim=0)
        value = torch.stack([_[3] for _ in data], dim=0)
        mode  = data[0][4]
        return h,r,t, value,mode

class OgOne2OneDataset(Dataset):
    def __init__(self, dp, nentity, nrelation, n_size=100,Datatype='Conf',pos_rat=1):
        self.nentity = nentity
        self.nrelation = nrelation
        self.Datatype = Datatype
        # 假设数据集都能够看到全部的
        if Datatype == 'Conf':
            self.pos_conf_triples,self.pos_sys_triples,self.pos_trans_triples = dp.getConSetData()
            trans_num =  int(pos_rat*len(self.pos_trans_triples))
            self.triples = self.pos_trans_triples[:trans_num] + self.pos_sys_triples
            self.triple_set = set(dp.trained)
            self.true_value = self.get_true_tails(-1,n_size+1)
        elif Datatype == 'Func':
            self.vio_triples, self.asy_triples,self.functional_triples = dp.getVioData()
            self.triples = self.functional_triples 
            self.triple_set = set(dp.trained)
            self.fun_rel_id = set(dp.fun_rel_idset)
            self.true_value = self.get_true_tails(-1.2,n_size+1)
        elif Datatype == 'Asy':
            self.vio_triples, self.asy_triples,self.functional_triples = dp.getVioData()
            self.triple_set = set(dp.trained)
            self.triples =self.asy_triples
            self.true_value = self.get_true_tails(-1.2,2)
        else:
            raise ValueError('Do not support thie data type Value: %s' % Datatype)
     
        self.n_size = n_size
        self.pofHead = OgOne2OneDataset.count_relation_frequency(self.triple_set,nrelation)
        self.true_head, self.true_tail = self.get_true_head_and_tail(self.triple_set)
        self.head_num = self.tail_num = nentity
        
    def __len__(self):
        return len(self.triples)
        
    def get_true_tails(self,init_value, size):
        true_value =  torch.zeros(size) 
        nn.init.constant_(true_value, init_value)
        true_value[0] = 1
        return true_value 

    def asysItem(self, idx):
        head,relation,tail =self.triples[idx]
        samples_h =torch.LongTensor( [head, tail])
        samples_t = torch.LongTensor([tail, head])
        samples_r = torch.LongTensor([relation,relation])
        shuffle_idx = torch.randperm(samples_h.nelement())
        return samples_h[shuffle_idx],samples_r[shuffle_idx], samples_t[shuffle_idx], self.true_value[shuffle_idx]
        

    def confAndFuncationalItem(self, idx):
        head,relation,tail =self.triples[idx]
        # samples = torch.zeros(self.n_size+1, 3,dtype=torch.long)
        sample_r = torch.LongTensor([relation]).expand(self.n_size+1)
        pr = self.pofHead[relation]
        rand_value = np.random.randint(np.iinfo(np.int32).max) % 1000
        
        if rand_value > pr:
            sample_h = torch.LongTensor([head]).expand(self.n_size+1)
            n_list = [[tail]]
        else:
            sample_t = torch.LongTensor([tail]).expand(self.n_size+1)
            n_list = [[head]]
        n_size = 0
        replace_index = 0
        while n_size < self.n_size:
            rdm_words = np.random.randint(0, self.nentity, self.n_size*2)
            if rand_value > pr:
                mask = np.in1d(
                    rdm_words, 
                    self.true_tail[(head, relation)], 
                    assume_unique=True, 
                    invert=True
                )
            else: # 如果小于则替换头实体
                mask = np.in1d(
                    rdm_words, 
                    self.true_head[(relation, tail)], 
                    assume_unique=True, 
                    invert=True
                )
            rdm_words = rdm_words[mask] # filter true triples
            n_list.append(rdm_words)
            n_size += rdm_words.size
        
        negative_sample = np.concatenate(n_list)[:self.n_size+1]

        if rand_value > pr:
            sample_t =  torch.LongTensor(negative_sample)
        else:
            sample_h = torch.LongTensor(negative_sample)

        shuffle_idx = torch.randperm(sample_h.nelement())

        return sample_h[shuffle_idx],sample_r[shuffle_idx], sample_t[shuffle_idx], self.true_value[shuffle_idx]
    
    def __getitem__(self, idx):
        if self.Datatype != 'Asy':
            return self.confAndFuncationalItem(idx)
        return self.asysItem(idx)
    
    @staticmethod
    def collate_fn(data):
        h = torch.stack([_[0] for _ in data], dim=0)
        r = torch.stack([_[1] for _ in data], dim=0)
        t = torch.stack([_[2] for _ in data], dim=0)
        value = torch.stack([_[3] for _ in data], dim=0)
        return h,r,t, value
    @staticmethod
    def count_relation_frequency(triples, nrelation):
        '''
        Get frequency of a partial triple like (head, relation) or (relation, tail)
        The frequency will be used for subsampling like word2vec
        '''
        tr2h = {}

        hr2t = {}
        for head, relation, tail in triples:
            if (head, relation) not in hr2t:
                hr2t[(head, relation)] = 1
            else:
                hr2t[(head, relation)] += 1

            if (tail,relation) not in tr2h:
                tr2h[(tail, relation)] = 1
            else:
                tr2h[(tail, relation)] += 1
        hrpt = {}  # 记录关系中每个头实体的平均尾实体数量
        trph = {}  # 记录关系中每个尾实体的平均头实体数量
        pofHead = {}
        for rid in range(nrelation):
            t_total =np.sum([hr2t[key] for key in hr2t.keys()])
            h_total = len(list(hr2t.keys()))
            hrpt[rid] = float(t_total)/h_total

            right_count =np.sum([tr2h[key] for key in tr2h.keys() ])
            left_count = len(list(tr2h.keys()))
            trph[rid] = float(right_count)/left_count
            pofHead[rid] = 1000* hrpt[rid]/(hrpt[rid]+trph[rid])
        return pofHead

    @staticmethod
    def get_true_head_and_tail(triples):
        '''
        Build a dictionary of true triples that will
        be used to filter these true triples for negative sampling
        '''
        
        true_head = {}
        true_tail = {}

        for head, relation, tail in triples:
            if (head, relation) not in true_tail:
                true_tail[(head, relation)] = []
            true_tail[(head, relation)].append(tail)
            if (relation, tail) not in true_head:
                true_head[(relation, tail)] = []
            true_head[(relation, tail)].append(head)

        for relation, tail in true_head:
            true_head[(relation, tail)] = np.array(list(set(true_head[(relation, tail)])))
        for head, relation in true_tail:
            true_tail[(head, relation)] = np.array(list(set(true_tail[(head, relation)])))                 

        return true_head, true_tail


class SampleOne2NDataset(Dataset):
    def __init__(self, normal,revesse, nentity, nrelation,norm_init,reverse_init,n_init=0, mode='hr_all', init_value = 1):
        self.nentity = nentity
        self.nrelation = nrelation

        norm_value = [norm_init for i in range(len(normal))]
        reverse_value = [reverse_init for i in range(len(revesse)) ]
        # norm_value.extend(reverse_value)
        # normal.extend(revesse)
        self.n_init = n_init
        self.triples = normal
        self.value = norm_value
        
        self.data_full,self.triple_value= self.build_dataset(self.triples,self.value)
        self.data =list(self.data_full.keys())

        self.mode = mode
        # 初始化entity2type
        self.init_value = init_value

    def build_dataset(self,triples,value):
        data = {}
        data2value = {}
        for i in range(len(triples)):
            h,r,t = triples[i]
            if not (h,r) not in data:
                data[(h,r)].append(t)
                data2value[(h,r)].append(value[i])
            else:
                data[(h,r)] = [t]
                data2value[(h,r)] = [value[i]]
        return data, data2value
    
    def __len__(self):
        return len(self.data)
        
    def get_true_tails(self,h_and_rs):
        true_tails =  torch.zeros(self.nentity,dtype=torch.float32)  
        nn.init.constant_(true_tails,self.n_init)
        for i in range(len(self.data_full[h_and_rs])):
            t = self.data_full[h_and_rs][i]
            true_tails[t] = self.triple_value[h_and_rs][i]
        return true_tails

    def __getitem__(self, idx):
        h_and_rs =self.data[idx]
        true_tails = self.get_true_tails(h_and_rs)
        h_and_rs = torch.tensor(h_and_rs)
        return h_and_rs, true_tails, self.mode
    
    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        # subsample_weight = torch.stack([_[2] for _ in data], dim=0)
        mode = data[0][2]
        # return positive_sample, negative_sample, subsample_weight, mode
        return positive_sample, negative_sample,  mode

class NewOne2OneDataset(Dataset):
    def __init__(self, triples, nentity, nrelation, init_value=-1,n_size=100, random=True, head_num = None, tail_num=None):
        self.nentity = nentity
        self.nrelation = nrelation
        self.triples = triples
        self.init_value = init_value
        self.triple_set = set(triples)
        self.n_size = n_size
        self.random = random
        self.pofHead = NewOne2OneDataset.count_relation_frequency(self.triple_set,nrelation)
        self.true_head, self.true_tail = self.get_true_head_and_tail(self.triple_set)
        if head_num == None:
            self.head_num = self.tail_num = nentity
        else:
            self.head_num = head_num
            self.tail_num = tail_num
        self.true_value = self.get_true_tails()
   
    def __len__(self):
        return len(self.triples)
        
    def get_true_tails(self):
        true_value =  torch.zeros(self.n_size+1) 
        nn.init.constant_(true_value, self.init_value)
        true_value[0] = 1
        return true_value 

    def __getitem__(self, idx):
        head,relation,tail =self.triples[idx]
        sample_r = torch.LongTensor([relation]).expand(self.n_size+1)
        pr = self.pofHead[relation]
        rand_value = np.random.randint(np.iinfo(np.int32).max) % 1000
        
        if rand_value > pr:
            sample_h = torch.LongTensor([head]).expand(self.n_size+1)
            n_list = [[tail]]
        else:
            sample_t = torch.LongTensor([tail]).expand(self.n_size+1)
            n_list = [[head]]
        n_size = 0
        replace_index = 0
        while n_size < self.n_size:

            if rand_value > pr: # 替换尾实体
                rdm_words = np.random.randint(0, self.tail_num, self.n_size*2)
                mask = np.in1d(
                    rdm_words, 
                    self.true_tail[(head, relation)], 
                    assume_unique=True, 
                    invert=True
                )
            else: # 如果小于则替换头实体
                rdm_words = np.random.randint(0, self.head_num, self.n_size*2)
                mask = np.in1d(
                    rdm_words, 
                    self.true_head[(relation, tail)], 
                    assume_unique=True, 
                    invert=True
                )
            rdm_words = rdm_words[mask] # filter true triples
            n_list.append(rdm_words)
            n_size += rdm_words.size
        
        negative_sample = np.concatenate(n_list)[:self.n_size+1]

        if rand_value > pr:
            sample_t =  torch.LongTensor(negative_sample)
        else:
            sample_h = torch.LongTensor(negative_sample)

        if self.random:
            shuffle_idx = torch.randperm(sample_h.nelement())
            return sample_h[shuffle_idx],sample_r[shuffle_idx], sample_t[shuffle_idx], self.true_value[shuffle_idx]
        else:
            return sample_h ,sample_r, sample_t, self.true_value
    
    @staticmethod
    def collate_fn(data):
        h = torch.stack([_[0] for _ in data], dim=0)
        r = torch.stack([_[1] for _ in data], dim=0)
        t = torch.stack([_[2] for _ in data], dim=0)
        value = torch.stack([_[3] for _ in data], dim=0)
        return h,r,t, value
    @staticmethod
    def count_relation_frequency(triples, nrelation):
        '''
        Get frequency of a partial triple like (head, relation) or (relation, tail)
        The frequency will be used for subsampling like word2vec
        '''
        tr2h = {}

        hr2t = {}
        for head, relation, tail in triples:
            if (head, relation) not in hr2t:
                hr2t[(head, relation)] = 1
            else:
                hr2t[(head, relation)] += 1

            if (tail,relation) not in tr2h:
                tr2h[(tail, relation)] = 1
            else:
                tr2h[(tail, relation)] += 1
        hrpt = {}  # 记录关系中每个头实体的平均尾实体数量
        trph = {}  # 记录关系中每个尾实体的平均头实体数量
        pofHead = {}
        for rid in range(nrelation):
            t_total =np.sum([hr2t[key] for key in hr2t.keys()])
            h_total = len(list(hr2t.keys()))
            hrpt[rid] = float(t_total)/h_total

            right_count =np.sum([tr2h[key] for key in tr2h.keys() ])
            left_count = len(list(tr2h.keys()))
            trph[rid] = float(right_count)/left_count
            pofHead[rid] = 1000* hrpt[rid]/(hrpt[rid]+trph[rid])
        return pofHead

    @staticmethod
    def get_true_head_and_tail(triples):
        '''
        Build a dictionary of true triples that will
        be used to filter these true triples for negative sampling
        '''
        
        true_head = {}
        true_tail = {}

        for head, relation, tail in triples:
            if (head, relation) not in true_tail:
                true_tail[(head, relation)] = []
            true_tail[(head, relation)].append(tail)
            if (relation, tail) not in true_head:
                true_head[(relation, tail)] = []
            true_head[(relation, tail)].append(head)

        for relation, tail in true_head:
            true_head[(relation, tail)] = np.array(list(set(true_head[(relation, tail)])))
        for head, relation in true_tail:
            true_tail[(head, relation)] = np.array(list(set(true_tail[(head, relation)])))                 

        return true_head, true_tail

class OGOneToOneTestDataset(Dataset):
    def __init__(self, triples, all_true_triples,nentity, nrelation, mode='hr_t'):
        self.nentity = nentity
        self.nrelation = nrelation
        self.triples = triples
        self.triple_set = set(all_true_triples)
        self.mode = mode

        if mode == 'hr_t':
            self.replace_index = 2
        elif mode == 'h_rt':
            self.replace_index = 0

    def __len__(self):
        return len(self.triples)
        

    def __getitem__(self, idx):
        head,relation,tail =self.triples[idx]
        samples = torch.zeros(self.nentity, 3,dtype=torch.long)
        samples = samples + torch.LongTensor((self.triples[idx])) 
        if self.mode == 'h_rt':
            tmp = [(0, rand_head) if (rand_head, relation, tail) not in self.triple_set
                   else (-1, head) for rand_head in range(self.nentity)]
          
            tmp[head] = (0, head)
        elif self.mode == 'hr_t':
            tmp = [(0, rand_tail) if (head, relation, rand_tail) not in self.triple_set
                   else (-1, tail) for rand_tail in range(self.nentity)]
            tmp[tail] = (0, tail)

        tmp  = torch.LongTensor(tmp)
        samples[...,self.replace_index] = tmp[...,1]
        filter_bias = tmp[...,0].float()
        postive_sampe = torch.LongTensor(self.triples[idx])
        return postive_sampe,samples,filter_bias,self.mode

    @staticmethod
    def collate_fn(data):
        positive = torch.stack([_[0] for _ in data], dim=0)
        samples = torch.stack([_[1] for _ in data], dim=0)
        filter_bias = torch.stack([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive, samples, filter_bias, mode

# 暂时用于ConvKB：h,r,t 的shape 应该是相同的
class OGOneToOneDataset(Dataset):
    def __init__(self, triples, nentity, nrelation, mode='hr_t',init_value=-1,n_size=100):
        self.nentity = nentity
        self.nrelation = nrelation
        self.triples = triples
        self.init_value = init_value
        self.triple_set = set(triples)
        self.mode = mode
        self.n_size = n_size
        self.count = self.count_frequency(self.triple_set)
        self.true_head, self.true_tail = self.get_true_head_and_tail(self.triple_set)
        self.head_num = self.tail_num = nentity
        if mode == 'hr_t':
            self.replace_index = 2
        elif mode == 'h_rt':
            self.replace_index = 0
   
    def __len__(self):
        return len(self.triples)
        
    def get_true_tails(self):
        true_value =  torch.zeros(self.n_size+1) 
        nn.init.constant_(true_value, self.init_value)
        true_value[0] = 1
        return true_value #,value

    def __getitem__(self, idx):
        head,relation,tail =self.triples[idx]
        subsampling_weight = self.count[(head, relation)] + self.count[(tail, -relation-1)]
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))  
        negative_sample_size=0
        negative_sample_list=[]

        samples = torch.zeros(self.n_size+1, 3,dtype=torch.long)
        samples = samples + torch.LongTensor((self.triples[idx]))
        while negative_sample_size < self.n_size:
            
            if self.mode == 'h_rt':
                negative_sample = np.random.randint(self.head_num, size=self.n_size*2)
                mask = np.in1d(
                    negative_sample, 
                    self.true_head[(relation, tail)], 
                    assume_unique=True, 
                    invert=True
                )
            elif self.mode == 'hr_t':
                negative_sample = np.random.randint(self.tail_num, size=self.n_size*2)
                mask = np.in1d(
                    negative_sample, 
                    self.true_tail[(head, relation)], 
                    assume_unique=True, 
                    invert=True
                )
            else:
                raise ValueError('Training batch mode %s not supported' % self.mode)
            
            negative_sample = negative_sample[mask] # filter true triples
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size

        negative_sample = np.concatenate(negative_sample_list)[:self.n_size]
       
        samples[1:,self.replace_index] =torch.LongTensor(negative_sample) 
        value = self.get_true_tails()
        return samples, value, subsampling_weight, self.mode

    
    @staticmethod
    def collate_fn(data):
        samples = torch.stack([_[0] for _ in data], dim=0)
        value = torch.stack([_[1] for _ in data], dim=0)
        subsample_weight = torch.stack([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return samples, value, subsample_weight, mode
    @staticmethod
    def count_frequency(triples, start=4):
        '''
        Get frequency of a partial triple like (head, relation) or (relation, tail)
        The frequency will be used for subsampling like word2vec
        '''
        count = {}
        for head, relation, tail in triples:
            if (head, relation) not in count:
                count[(head, relation)] = start
            else:
                count[(head, relation)] += 1

            if (tail, -relation-1) not in count:
                count[(tail, -relation-1)] = start
            else:
                count[(tail, -relation-1)] += 1
        return count
    
    @staticmethod
    def get_true_head_and_tail(triples):
        '''
        Build a dictionary of true triples that will
        be used to filter these true triples for negative sampling
        '''
        
        true_head = {}
        true_tail = {}

        for head, relation, tail in triples:
            if (head, relation) not in true_tail:
                true_tail[(head, relation)] = []
            true_tail[(head, relation)].append(tail)
            if (relation, tail) not in true_head:
                true_head[(relation, tail)] = []
            true_head[(relation, tail)].append(head)

        for relation, tail in true_head:
            true_head[(relation, tail)] = np.array(list(set(true_head[(relation, tail)])))
        for head, relation in true_tail:
            true_tail[(head, relation)] = np.array(list(set(true_tail[(head, relation)])))                 

        return true_head, true_tail

class OGNagativeSampleDataset(Dataset):
    def __init__(self, dp,all_trained_triple, nentity, nrelation, negative_sample_size, mode,pos_rat=1,Datatype='Conf',head_num=None,tail_num=None):
        # 假设数据集都能够看到全部的
        if Datatype == 'Conf':
            self.pos_conf_triples,self.pos_sys_triples,self.pos_trans_triples = dp.getConSetData()
            trans_num =  int(pos_rat*len(self.pos_trans_triples))
            self.triples = self.pos_trans_triples[:trans_num] + self.pos_sys_triples
            self.triple_set = set(all_trained_triple)
        elif Datatype == 'Func':
            self.vio_triples, self.asy_triples,self.functional_triples = dp.getVioData()
            self.triples = self.functional_triples 
            self.triple_set = all_trained_triple
            self.fun_rel_id = set(dp.fun_rel_idset)
        elif Datatype == 'Asy':
            self.vio_triples, self.asy_triples,self.functional_triples = dp.getVioData()
            self.triple_set = all_trained_triple
            self.triples =self.asy_triples
        else:
            raise ValueError('Do not support thie data type Value: %s' % Datatype)

        self.dataType = Datatype
        self.len = len(self.triples)

        self.nentity = nentity
        self.nrelation = nrelation

        self.negative_sample_size = negative_sample_size
        self.mode = mode

        self.count = self.count_frequency(self.triple_set)
        self.true_head, self.true_tail = self.get_true_head_and_tail(self.triple_set)

        if not head_num is None and not tail_num is None:
            self.head_num = head_num
            self.tail_num = tail_num
        elif head_num is None and tail_num is None:
            self.head_num = self.tail_num = self.nentity
        else:
            print("ERROR")

    def __len__(self):
        return self.len
    
    def getItem_conf(self,idx):
        positive_sample = self.triples[idx]
        head, relation, tail = positive_sample
        subsampling_weight = self.count[(head, relation)] + self.count[(tail, -relation-1)]
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))
        
        negative_sample_list = []
        negative_sample_size = 0

        while negative_sample_size < self.negative_sample_size:
            
            if self.mode == 'h_rt':
                negative_sample = np.random.randint(self.head_num, size=self.negative_sample_size*2)
                mask = np.in1d(
                    negative_sample, 
                    self.true_head[(relation, tail)], 
                    assume_unique=True, 
                    invert=True
                )
            elif self.mode == 'hr_t':
                negative_sample = np.random.randint(self.tail_num, size=self.negative_sample_size*2)
                mask = np.in1d(
                    negative_sample, 
                    self.true_tail[(head, relation)], 
                    assume_unique=True, 
                    invert=True
                )
            else:
                raise ValueError('Training batch mode %s not supported' % self.mode)
            
            negative_sample = negative_sample[mask] # filter true triples
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size

        negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]
        negative_sample = torch.LongTensor(negative_sample)
        positive_sample = torch.LongTensor(positive_sample)

        return positive_sample, negative_sample, subsampling_weight, self.mode

    def getItemVio(self, idx):

        positive_sample = self.triples[idx]
        head, relation, tail = positive_sample

        subsampling_weight = self.count[(head, relation)] + self.count[(tail, -relation-1)]
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))
        negative_sample = [tail,relation,head]
        negative_sample = torch.LongTensor(negative_sample)
        positive_sample = torch.LongTensor(positive_sample)
        return positive_sample, negative_sample, subsampling_weight, self.mode       

    def __getitem__(self, idx):
        if self.dataType  != 'Asy':
            return self.getItem_conf(idx)
        else:
            return self.getItemVio(idx)
    
    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        subsample_weight = torch.cat([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, subsample_weight, mode
    
    @staticmethod
    def count_frequency(triples, start=4):
        '''
        Get frequency of a partial triple like (head, relation) or (relation, tail)
        The frequency will be used for subsampling like word2vec
        '''
        count = {}
        for head, relation, tail in triples:
            if (head, relation) not in count:
                count[(head, relation)] = start
            else:
                count[(head, relation)] += 1

            if (tail, -relation-1) not in count:
                count[(tail, -relation-1)] = start
            else:
                count[(tail, -relation-1)] += 1
        return count
    
    @staticmethod
    def get_true_head_and_tail(triples):
        '''
        Build a dictionary of true triples that will
        be used to filter these true triples for negative sampling
        '''
        
        true_head = {}
        true_tail = {}

        for head, relation, tail in triples:
            if (head, relation) not in true_tail:
                true_tail[(head, relation)] = []
            true_tail[(head, relation)].append(tail)
            if (relation, tail) not in true_head:
                true_head[(relation, tail)] = []
            true_head[(relation, tail)].append(head)

        for relation, tail in true_head:
            true_head[(relation, tail)] = np.array(list(set(true_head[(relation, tail)])))
        for head, relation in true_tail:
            true_tail[(head, relation)] = np.array(list(set(true_tail[(head, relation)])))                 

        return true_head, true_tail

class NagativeSampleNewDataset(Dataset):

    def splitByTypeDiff(self,typeDiff):
        resultDicts = []
        thread1 = 0.1
        thread2 = 0.6
        for i in range(len(typeDiff)):
            typeDict = {"small":[],"middle":[],"big":[]}
            for j in range(len(typeDiff[0])):
                if typeDiff[i][j] <= thread1:
                    typeDict['small'].append(j)
                elif typeDiff[i][j] >= thread2:
                    typeDict['big'].append(j)
                else:
                    typeDict['middle'].append(j)
            typeDict['small'] = np.array(typeDict['small'])
            typeDict['big'] = np.array(typeDict['big'])
            typeDict['middle'] = np.array(typeDict['middle'])
            resultDicts.append(typeDict)
        return resultDicts
    
    def read_type2entity(self):
        file = "/home/skl/yl/data/YAGO3-1668k/yago_insnet/type2entity.txt"
        type2Eids = []
        with open(file, 'r',encoding='utf-8') as fin:
            data = fin.readlines()
            for line in data:
                eids = line.strip().split('\t')
                type2Eids.append(list(map(int,eids)))
        return np.array(type2Eids,dtype=object)


    def buildType2entity(self):
        type2Eids = self.read_type2entity()
        resultDicts = self.splitByTypeDiff(self.typeDiff)
        splitType2eid = []
        for typeInfo in resultDicts:
            small = typeInfo['small']
            middle = typeInfo['middle']
            big = typeInfo['big']

            if len(small) > 0:
                small_type = type2Eids[small]
            else:
                small_type = np.array([[]])
            if len(middle) > 0:
                middle_type = type2Eids[middle]
            else:
                middle_type = np.array([[]])
            if len(big) > 0:
                big_type = type2Eids[big]
            else:
                big_type = np.array([[]])
            
            small_type = np.concatenate(small_type)
            middle_type = np.concatenate(middle_type)
            big_type = np.concatenate(big_type)
            splitType = {
                'small':small_type,
                'middle':middle_type,
                'big':big_type
            }
            splitType2eid.append(splitType)
        return splitType2eid
    
    def getitem_byType(self,idx):

        positive_sample = self.triples[idx]
        head, relation, tail = positive_sample
        subsampling_weight = self.count[(head, relation)] + self.count[(tail, -relation-1)]
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))

        negative_sample_list = []
        negative_sample_size = 0

        if self.mode =='h_rt':
            true_type_id = self.en2typeid[head]
        elif self.mode =='hr_t':
            true_type_id = self.en2typeid[tail]
        
        splitType = self.splitType2eid[true_type_id]


        small_type = splitType['small']
        middle_type = splitType['middle']
        big_type = splitType['big']
        samll_flag = True
        while negative_sample_size < self.negative_sample_size:
            if samll_flag:
                result = []
                if len(small_type) > 0:
                    samll = small_type[np.random.randint(len(small_type), size=self.small_size)].astype(
                        np.int64)
                    result.append(samll)
                if len(middle_type) > 0:
                    middle = middle_type[np.random.randint(len(middle_type), size=self.middle_size)].astype(
                        np.int64)
                    result.append(middle)
                if len(big_type) > 0:
                    big = big_type[np.random.randint(len(big_type), size=self.big_size)].astype(
                        np.int64)
                    result.append(big)
                negative_sample =  np.concatenate(result)
            else:
                negative_sample = np.random.randint(self.nentity, size=self.negative_sample_size*2)
            
            if self.mode == 'h_rt':
                mask = np.in1d(
                    negative_sample, 
                    self.true_head[(relation, tail)], 
                    assume_unique=True, 
                    invert=True
                )
            elif self.mode == 'hr_t':
                mask = np.in1d(
                    negative_sample, 
                    self.true_tail[(head, relation)], 
                    assume_unique=True, 
                    invert=True
                )
            else:
                raise ValueError('Training batch mode %s not supported' % self.mode)
            
            negative_sample = negative_sample[mask] # filter true triples
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size
            if  samll_flag :
                samll_flag = False
        negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]
        negative_sample = torch.LongTensor(negative_sample)
        positive_sample = torch.LongTensor(positive_sample)
        return positive_sample, negative_sample, subsampling_weight, self.mode


    # 类语义信息：disjoint 信息
    def __init__(self, triples, nentity, nrelation, negative_sample_size, mode, entity2type=None, typeDiff=None,en2typeid=None):
        self.len = len(triples)
        self.triples = triples
        self.triple_set = (triples)
        self.nentity = nentity
        self.type = "Disjoint"
        self.type = "TypeSub"

        self.small_size = int(negative_sample_size*0.1)
        self.middle_size = int( negative_sample_size*0.7)
        self.big_size = int(negative_sample_size*0.2)

        # 初始化entity2type
        if  entity2type != None:
            self.entity2tye = [i for i in range(nentity)]
            typeid = 0
            for type in entity2type.keys():
                for index in entity2type[type]:
                    self.entity2tye[index] = typeid
                typeid += 1
            self.entity2tye = np.array(self.entity2tye)
        
        
        self.en2typeid = np.array(en2typeid)
        self.typeDiff =np.array(typeDiff)

        # self.splitType2eid = self.buildType2entity()
        self.nrelation = nrelation
        self.negative_sample_size = negative_sample_size
        self.mode = mode
        self.count = self.count_frequency(triples)
        self.true_head, self.true_tail = self.get_true_head_and_tail(self.triples)

    def __len__(self):
        return self.len

    def getitem_with_weight(self, idx):
        positive_sample = self.triples[idx]
        head, relation, tail = positive_sample
        subsampling_weight = self.count[(head, relation)] + self.count[(tail, -relation-1)]
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))
        
        negative_sample_list = []
        negative_sample_size = 0
        nagative_weight_list = []

        if self.mode =='h_rt':
            true_type = self.entity2tye[head]
            true_type_id = self.en2typeid[head]
        elif self.mode =='hr_t':
            true_type = self.entity2tye[tail]
            true_type_id = self.en2typeid[tail]

        while negative_sample_size < self.negative_sample_size:
            negative_sample = np.random.randint(self.nentity, size=self.negative_sample_size*2)
            
            if self.mode == 'h_rt':
                mask = np.in1d(
                    negative_sample, 
                    self.true_head[(relation, tail)], 
                    assume_unique=True, 
                    invert=True
                )
            elif self.mode == 'hr_t':
                mask = np.in1d(
                    negative_sample, 
                    self.true_tail[(head, relation)], 
                    assume_unique=True, 
                    invert=True
                )
            else:
                raise ValueError('Training batch mode %s not supported' % self.mode)
            
            
            negative_sample = negative_sample[mask] # filter true triples

            if self.type == 'Disjoint':
                nagative_weight = [1 for i in range(len(negative_sample))] 
                nagative_weight = np.array(nagative_weight)
                typeids = self.entity2tye[negative_sample] != true_type
                nagative_weight[typeids] = 10
            else:
                typeIds = self.en2typeid[negative_sample]
                nagative_weight =  -(self.typeDiff[true_type_id][typeIds]-0.6)*2

            nagative_weight_list.append(nagative_weight)

            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size

        negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]
        nagative_weight = np.concatenate(nagative_weight_list)[:self.negative_sample_size]
        negative_sample = torch.LongTensor(negative_sample)
        nagative_weight = torch.FloatTensor(nagative_weight)
        positive_sample = torch.LongTensor(positive_sample)
        return positive_sample, negative_sample, subsampling_weight, self.mode,nagative_weight

    def __getitem__(self, idx):
        return self.getitem_with_weight(idx)
    
    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        subsample_weight = torch.cat([_[2] for _ in data], dim=0)
        weight_list = torch.stack([_[4] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, subsample_weight, mode ,weight_list
        # return positive_sample, negative_sample, subsample_weight, mode  # ,weight_list
    
    @staticmethod
    def count_frequency(triples, start=4):
        '''
        Get frequency of a partial triple like (head, relation) or (relation, tail)
        The frequency will be used for subsampling like word2vec
        '''
        count = {}
        for head, relation, tail in triples:
            if (head, relation) not in count:
                count[(head, relation)] = start
            else:
                count[(head, relation)] += 1

            if (tail, -relation-1) not in count:
                count[(tail, -relation-1)] = start
            else:
                count[(tail, -relation-1)] += 1
        return count
    
    @staticmethod
    def get_true_head_and_tail(triples):
        '''
        Build a dictionary of true triples that will
        be used to filter these true triples for negative sampling
        '''
        
        true_head = {}
        true_tail = {}

        for head, relation, tail in triples:
            if (head, relation) not in true_tail:
                true_tail[(head, relation)] = []
            true_tail[(head, relation)].append(tail)
            if (relation, tail) not in true_head:
                true_head[(relation, tail)] = []
            true_head[(relation, tail)].append(head)

        for relation, tail in true_head:
            true_head[(relation, tail)] = np.array(list(set(true_head[(relation, tail)])))
        for head, relation in true_tail:
            true_tail[(head, relation)] = np.array(list(set(true_tail[(head, relation)])))                 

        return true_head, true_tail

class NagativeSampleDataset(Dataset):
    def __init__(self, triples, nentity, nrelation, negative_sample_size, mode,head_num=None,tail_num=None,all_traied=None,head_begin=0,tail_begin=0):

        self.len = len(triples)
        self.triples = triples
        if all_traied != None:
            self.triple_set = (all_traied)
        else:
            self.triple_set = (triples)

        self.nentity = nentity
        self.nrelation = nrelation
        self.negative_sample_size = negative_sample_size
        self.mode = mode
        self.count = self.count_frequency(self.triple_set)


        self.true_head, self.true_tail = self.get_true_head_and_tail(self.triple_set)
        self.head_begin = head_begin
        self.tail_begin = tail_begin
        if not head_num is None and not tail_num is None:
            self.head_num = head_num
            self.tail_num = tail_num
        elif head_num is None and tail_num is None:
            self.head_num = self.tail_num = self.nentity
        else:
            print("ERROR")
        

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        positive_sample = self.triples[idx]
        head, relation, tail = positive_sample
        subsampling_weight = self.count[(head, relation)] + self.count[(tail, -relation-1)]
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))
        
        negative_sample_list = []
        negative_sample_size = 0

        while negative_sample_size < self.negative_sample_size:
            
            if self.mode == 'h_rt':
                negative_sample = np.random.randint(self.head_begin,self.head_num+self.head_begin, size=self.negative_sample_size*2)
                mask = np.in1d(
                    negative_sample, 
                    self.true_head[(relation, tail)], 
                    assume_unique=True, 
                    invert=True
                )
            elif self.mode == 'hr_t':
                negative_sample = np.random.randint(self.tail_begin,self.tail_num+self.tail_begin, size=self.negative_sample_size*2)
                mask = np.in1d(
                    negative_sample, 
                    self.true_tail[(head, relation)], 
                    assume_unique=True, 
                    invert=True
                )
            else:
                raise ValueError('Training batch mode %s not supported' % self.mode)
            
            negative_sample = negative_sample[mask] # filter true triples
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size

        negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]
        negative_sample = torch.LongTensor(negative_sample)
        positive_sample = torch.LongTensor(positive_sample)

        return positive_sample, negative_sample, subsampling_weight, self.mode
    
    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        subsample_weight = torch.cat([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, subsample_weight, mode
    
    @staticmethod
    def count_frequency(triples, start=4):
        '''
        Get frequency of a partial triple like (head, relation) or (relation, tail)
        The frequency will be used for subsampling like word2vec
        '''
        count = {}
        for head, relation, tail in triples:
            if (head, relation) not in count:
                count[(head, relation)] = start
            else:
                count[(head, relation)] += 1

            if (tail, -relation-1) not in count:
                count[(tail, -relation-1)] = start
            else:
                count[(tail, -relation-1)] += 1
        return count
    
    @staticmethod
    def get_true_head_and_tail(triples):
        '''
        Build a dictionary of true triples that will
        be used to filter these true triples for negative sampling
        '''
        
        true_head = {}
        true_tail = {}

        for head, relation, tail in triples:
            if (head, relation) not in true_tail:
                true_tail[(head, relation)] = []
            true_tail[(head, relation)].append(tail)
            if (relation, tail) not in true_head:
                true_head[(relation, tail)] = []
            true_head[(relation, tail)].append(head)

        for relation, tail in true_head:
            true_head[(relation, tail)] = np.array(list(set(true_head[(relation, tail)])))
        for head, relation in true_tail:
            true_tail[(head, relation)] = np.array(list(set(true_tail[(head, relation)])))                 

        return true_head, true_tail

class OneToNDataset(Dataset):
    def __init__(self, triples, nentity, nrelation, mode='hr_all', init_value = 1,entity2type=None, typeDiff=None,en2typeid=None):
        self.nentity = nentity
        self.nrelation = nrelation
        self.triples = triples
        self.data_full= self.build_dataset(triples)
        self.data =list(self.data_full.keys())
        self.mode = mode
        # 初始化entity2type
        self.init_value = init_value
        if  entity2type != None:
            self.entity2tye = [i for i in range(nentity)]
            typeid = 0
            for type in entity2type.keys():
                for index in entity2type[type]:
                    self.entity2tye[index] = typeid
                typeid += 1
            self.entity2tye = np.array(self.entity2tye)
            self.en2typeid = np.array(en2typeid).ravel()
            self.typeDiff = 1.5 - np.array(typeDiff)
            print(self.typeDiff.shape)
            self.all_typeId = self.en2typeid[[i for i in range(self.nentity)]]

    def build_dataset(self,triples):
        data = {}
        for h, r,t in triples:
            if not (h,r) not in data:
                data[(h,r)].append(t)
            else:
                data[(h,r)] = [t]
        return data
        
    def __len__(self):
        return len(self.data)
        
    def get_true_tails(self,h_and_rs):
        true_tails =  torch.zeros(self.nentity)  # 不同的会有不同的要求
        for t in self.data_full[h_and_rs]:
            true_tails[t] = self.init_value
        # truth_typeId = self.en2typeid[self.data_full[h_and_rs]]
        # value =torch.FloatTensor(self.typeDiff[truth_typeId[0]][...,self.all_typeId])
        return true_tails #,value

    def __getitem__(self, idx):
        h_and_rs =self.data[idx]
        true_tails = self.get_true_tails(h_and_rs)
        h_and_rs = torch.tensor(h_and_rs)
       # return h_and_rs, true_tails, value, self.mode
        return h_and_rs, true_tails, self.mode
    
    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        # subsample_weight = torch.stack([_[2] for _ in data], dim=0)
        mode = data[0][2]
        # return positive_sample, negative_sample, subsample_weight, mode
        return positive_sample, negative_sample,  mode

class SimpleTripleDataset(Dataset):
    def __init__(self, triples, all_true_triples, nentity, nrelation):
        self.nentity = nentity
        self.nrelation = nrelation
        self.all_true_triples = all_true_triples
        self.triples = triples
        self.data_full= self.build_dataset(all_true_triples)
        self.data =list(self.data_full.keys())

    def build_dataset(self,triples):
        data = {}
        for h, r,t in triples:
            if not (h,r) not in data:
                data[(h,r)].append(t)
            else:
                data[(h,r)] = [t]
        return data
        
    def __len__(self):
        return len(self.triples)
    
    def get_true_tails(self,h_and_rs):
        true_tails =  torch.zeros(self.nentity)
        for t  in self.data_full[h_and_rs]:
            true_tails[t] = 1
        return true_tails

    def __getitem__(self, idx):
        triple = self.triples[idx]
        h_and_rs = (triple[0],triple[1])
        true_tail = triple[2]
        filter = self.get_true_tails(h_and_rs)
        h_and_rs = torch.tensor(h_and_rs)
        true_tail = torch.tensor(true_tail)
        return h_and_rs, true_tail, filter

class TestDataset(Dataset):
    def __init__(self, triples, all_true_triples, nentity, nrelation, mode,head_num=None,tail_num=None):
        self.len = len(triples)
        self.triple_set = set(all_true_triples)
        self.triples = triples
        self.nentity = nentity
        self.nrelation = nrelation
        self.mode = mode

        if not head_num is None and not tail_num is None:
            self.head_num = head_num
            self.tail_num = tail_num
        elif head_num is None and tail_num is None:
            self.head_num = self.tail_num = self.nentity
        else:
            print("ERROR")

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        head, relation, tail = self.triples[idx]
        if self.mode == 'h_rt':
            tmp = [(0, rand_head) if ((rand_head, relation, tail) not in self.triple_set and rand_head != tail)
                   else (-1, head) for rand_head in range(self.head_num)]
          
            tmp[head] = (0, head)
        elif self.mode == 'hr_t':
            tmp = [(0, rand_tail) if ((head, relation, rand_tail) not in self.triple_set and rand_tail != head)
                   else (-1, tail) for rand_tail in range(self.tail_num)]
            tmp[tail] = (0, tail)
        else:
            raise ValueError('negative batch mode %s not supported' % self.mode)
            
        tmp = torch.LongTensor(tmp)            
        filter_bias = tmp[:, 0].float()
        negative_sample = tmp[:, 1]
        positive_sample = torch.LongTensor((head, relation, tail))
        return positive_sample, negative_sample, filter_bias, self.mode
    
    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        filter_bias = torch.stack([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, filter_bias, mode

# 包含多个triples 组合的数据集，用于关系分类测试
class MulTestDataset(Dataset):

    def __init__(self, triples, all_true_triples, nentity, nrelation, mode="hr_t"):
        self.len = len(triples[0])

        self.triples_num = len(triples)
        self.triple_set = set(all_true_triples)
        self.triples = triples
        self.nentity = nentity
        self.nrelation = nrelation
        self.mode = mode


    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        positive_samples = []
        negative_samples = []
        filter_biass = []
        for i in range(self.triples_num):
            head, relation, tail = self.triples[i][idx]
            if self.mode == 'h_rt':
                tmp = [(0, rand_head) if (rand_head, relation, tail) not in self.triple_set
                    else (-1, head) for rand_head in range(self.nentity)]
                tmp[head] = (0, head)
            elif self.mode == 'hr_t':
                tmp = [(0, rand_tail) if (head, relation, rand_tail) not in self.triple_set
                    else (-1, tail) for rand_tail in range(self.nentity)]
                tmp[tail] = (0, tail)
            else:
                raise ValueError('negative batch mode %s not supported' % self.mode)
            tmp = torch.LongTensor(tmp)            
            filter_bias = tmp[:, 0].float()
            negative_sample = tmp[:, 1]
            positive_sample = torch.LongTensor((head, relation, tail))
            positive_samples.append(positive_sample)
            negative_samples.append(negative_sample)
            filter_biass.append(filter_bias)
        positive_samples = torch.stack(list(positive_samples), dim=0)
        negative_samples = torch.stack(list(negative_samples), dim=0)
        filter_biass = torch.stack(list(filter_biass), dim=0)
        return  positive_samples, negative_samples, filter_biass, self.mode
    
    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        filter_bias = torch.stack([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, filter_bias, mode

# 针对两个数据集的构造迭代器
class BidirectionalOneShotIterator(object):
    def __init__(self, dataloader_head, dataloader_tail):
        self.iterator_head = self.one_shot_iterator(dataloader_head)
        self.iterator_tail = self.one_shot_iterator(dataloader_tail)
        self.len = len(dataloader_head) + len(dataloader_tail)
        self.step = 0
        self.epoch = 0
        
    def __next__(self):
        self.step += 1
        if(self.step //self.len) > self.epoch:
            self.epoch += 1
        if self.step % 2 == 0:
            data = next(self.iterator_head)
        else:
            data = next(self.iterator_tail)
        return data
    
    @staticmethod
    def one_shot_iterator(dataloader):
        '''
        Transform a PyTorch Dataloader into python iterator
        '''
        while True:
            for data in dataloader:
                yield data


class MultiShotItertor(object):
    def __init__(self, a,b,c,d):
        self.a = self.one_shot_iterator(a)
        self.b = self.one_shot_iterator(b)
        self.c = self.one_shot_iterator(c)
        self.d = self.one_shot_iterator(d)
        self.step = 0

    def __next__(self):
        
        if self.step % 4 == 0:
            data = next(self.a)
        elif self.step % 4 == 1:
            data = next(self.b)
        elif self.step % 4 == 2:
            data = next(self.c)
        elif self.step % 4 == 3:
            data = next(self.d)
        self.step += 1
        return data

    @staticmethod
    def one_shot_iterator(dataloader):
        '''
        Transform a PyTorch Dataloader into python iterator
        '''
        while True:
            for data in dataloader:
                yield data
                
# 针对一个数据集构造迭代器
class OneShotIterator(object):
    def __init__(self, dataloader):
        self.dataloader = self.one_shot_iterator(dataloader)
 
    def __next__(self):
        data = next(self.dataloader)
        return data
    
    @staticmethod
    def one_shot_iterator(dataloader):
        '''
        Transform a PyTorch Dataloader into python iterator
        '''
        while True:
            for data in dataloader:
                yield data
