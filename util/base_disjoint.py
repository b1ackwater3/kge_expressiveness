import os 
from collections import defaultdict
from torch.utils.data import Dataset
import torch
import numpy as np
import torch.nn as nn

class DisjointProcesser:

    def __init__(self,data_path):
        super(DisjointProcesser, self).__init__()

        self.data_path = data_path
        self.entity2id,self.relation2id = None, None
        
        self.entity2id,self.relation2id = DisjointProcesser.read_idDic(self.data_path)
        self.nentity = len(self.entity2id)
        self.nrelation = len(self.relation2id)
       
        self.id2relation = {
                self.relation2id[k]:k for k in self.relation2id.keys()
            }
        self.id2entity = {
                self.entity2id[k]:k for k in self.entity2id.keys()
            }
        self.read_normal_data(False)

        h2t = defaultdict(set)
        for h,r,t in self.train:
            h2t[h].add((t))
        
        train_set = set(self.train)
        
        for h,r,t in self.train:
            for t2 in h2t[t]:
                if (h,r,t2) not in train_set:
                    train_set.add((h,r,t2))
        
        self.train = list(train_set)


    def build_transfor_test_filter(self):
        all_true =set(self.all_true_triples)
        h2ts = defaultdict(set)
        for h,r,t in all_true:
            h2ts[h].add(t)
        
        trans_close = set()
        
        for h,r,t in all_true:
            trans_close.add((h,r,t))
            for second_t in h2ts[t]:
                if (h,r,second_t) not in all_true:
                    trans_close.add((h,r,second_t))

        self.all_true_triples = trans_close

    def get_classtest_list(self):
        if self.dataType != "ClassTest":
            raise ValueError('Only ClassTest Dataset can get ClassTest List')
        [self.class_test_data[relation_type] for relation_type in ['symmetry','asymmetry','inverse','transitive','composition']]

    def read_normal_data(self,is_id):
        self.train = DisjointProcesser.read_triples(os.path.join(self.data_path,'train.txt'),self.entity2id,self.relation2id,is_id=is_id)
        self.test =  DisjointProcesser.read_triples(os.path.join(self.data_path,'test.txt'),self.entity2id,self.relation2id,is_id=is_id)
        self.valid = DisjointProcesser.read_triples(os.path.join(self.data_path,'valid.txt'),self.entity2id,self.relation2id,is_id=is_id)
        self.all_true_triples = self.train + self.valid + self.test

    @staticmethod
    def read_idDic(data_path):
        with open(os.path.join(data_path, 'entities.dict'),encoding='utf-8') as fin:
            entity2id = dict()
            for line in fin:
                eid, entity = line.strip().split('\t')
                entity2id[entity] = int(eid)
        with open(os.path.join(data_path, 'relations.dict'),encoding='utf-8') as fin:
            relation2id = dict()
            for line in fin:
                rid, relation = line.strip().split('\t')
                relation2id[relation] = int(rid)
        return entity2id,relation2id

    @staticmethod
    def read_triples(file_path, entity2id=None, relation2id=None, is_id=False, distinct = True):
        if distinct:
            triples = set()
        else:
            triples = []
        is_tran = False
        if entity2id is not None and relation2id is not None:
            is_tran = True
        elif entity2id is None and relation2id is None:
            is_tran = False
        else:
            raise ValueError("entity2id and relation2id should both be None or not be None")
        with open(file_path, encoding='utf-8') as fin:
            for line in fin:
                h, r, t = line.strip().split('\t')  
                # if h == t:           # 去掉自反的数据
                #     continue
                if distinct:
                    if is_id:
                        triples.add((int(h),int(r),int(t)))
                    elif is_tran:
                        if h not in entity2id or t not in entity2id or r not in relation2id:
                            continue
                        triples.add((entity2id[h], relation2id[r], entity2id[t]))
                    else:
                        triples.add((h,r,t))
                else:
                    if is_id:
                        triples.append((int(h),int(r),int(t)))
                    elif is_tran:
                        triples.append((entity2id[h], relation2id[r], entity2id[t]))
                    else:
                        triples.append((h,r,t))
        return list(triples)

    @staticmethod
    def build_trans_triple(triples, hr2triples, rt2triples, filter):
        new_build_triples = []
        triples = set(triples)
        print("Trans Caculate begin")
        new_triples = set()
        for h,r,t in triples:
            new_triples.add((h,r,t))
            for triple2 in hr2triples[(t,r)]:
                if (h,r,triple2[2]) not in triples and (h,r,triple2[2]) not in new_triples and \
                (h,r,triple2[2]) not in filter:
                    new_triples.add((h,r,triple2[2]))
                    new_build_triples.append((h,r,triple2[2]))
            for triple2 in rt2triples[(r,h)]:
                if (triple2[0],r,t) not in triples and (triple2[0],r,t) not in new_triples \
                and (triple2[0],r,t) not in filter:
                    new_triples.add((triple2[0],r,t))
                    new_build_triples.append((triple2[0],r,t))
        triples = new_triples

        return new_build_triples


class NagativeSampleDisjointDataset(Dataset):
    def __init__(self, triples, disjoint_triples, nentity, nrelation, negative_sample_size, mode):

        self.h2disjoint = defaultdict(list)
        self.t2disjoint = defaultdict(list)

        for h,r,t in disjoint_triples:
            self.h2disjoint[h].append(t)
            self.t2disjoint[t].append(h)

        for k in self.h2disjoint.keys():
            self.h2disjoint[k] = np.array(self.h2disjoint[k])

        for k in self.t2disjoint.keys():
            self.t2disjoint[k] = np.array(self.t2disjoint[k])
      

        filtered_triple = []
        for h,r,t in triples:
            if mode == "hr_t":
                if h in set(self.h2disjoint.keys()):
                    filtered_triple.append((h,r,t))
            else:
                if t in set(self.t2disjoint.keys()):
                    filtered_triple.append((h,r,t))
            
        filtered_triple = list(set(filtered_triple))
        self.len = len(filtered_triple)
        self.triples = filtered_triple
      
        self.triple_set = set(filtered_triple)

        self.nentity = nentity
        self.nrelation = nrelation
        self.negative_sample_size = negative_sample_size
        self.mode = mode
        self.count = self.count_frequency(self.triple_set)
        self.true_head, self.true_tail = self.get_true_head_and_tail(self.triple_set)

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        positive_sample = self.triples[idx]
        head, relation, tail = positive_sample
     
        negative_sample_list = []

        if self.mode == 'h_rt':
            negative_sample =(np.random.randint(0,len(self.t2disjoint[tail]), size=self.negative_sample_size))
            negative_sample_list =  self.t2disjoint[tail][negative_sample]
        elif self.mode == 'hr_t':
            negative_sample =(np.random.randint(0,len(self.h2disjoint[head]), size=self.negative_sample_size))
            negative_sample_list =  self.h2disjoint[head][negative_sample]
        else:
            raise ValueError('Training batch mode %s not supported' % self.mode)
            

        negative_sample = negative_sample_list
        negative_sample = torch.LongTensor(negative_sample)
        positive_sample = torch.LongTensor(positive_sample)

        return positive_sample, negative_sample,  self.mode
    
    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        mode = data[0][2]
        return positive_sample, negative_sample , mode
    
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

class One2OneDisjointDataset(Dataset):
    def __init__(self, triples, disjoint_triples, nentity, nrelation, negative_sample_size, mode):

        self.h2disjoint = defaultdict(list)
        self.t2disjoint = defaultdict(list)

        for h,r,t in disjoint_triples:
            self.h2disjoint[h].append(t)
            self.t2disjoint[t].append(h)

        for k in self.h2disjoint.keys():
            self.h2disjoint[k] = np.array(self.h2disjoint[k])

        for k in self.t2disjoint.keys():
            self.t2disjoint[k] = np.array(self.t2disjoint[k])
      

        filtered_triple = []
        for h,r,t in triples:
            if mode == "hr_t":
                if h in set(self.h2disjoint.keys()):
                    filtered_triple.append((h,r,t))
            else:
                if t in set(self.t2disjoint.keys()):
                    filtered_triple.append((h,r,t))
            
        filtered_triple = list(set(filtered_triple))
        self.len = len(filtered_triple)
        self.triples = filtered_triple
      
        self.triple_set = set(filtered_triple)

        self.nentity = nentity
        self.nrelation = nrelation
        self.negative_sample_size = negative_sample_size
        self.mode = mode
        self.count = self.count_frequency(self.triple_set)
        self.true_head, self.true_tail = self.get_true_head_and_tail(self.triple_set)
        self.true_value = self.get_true_tails()
        
    def get_true_tails(self):
        true_value =  torch.zeros(self.negative_sample_size+1) 
        nn.init.constant_(true_value, -1.2)
        true_value[0] = 1
        return true_value 
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        head,relation,tail =self.triples[idx]
        sample_r = torch.LongTensor([relation]).expand(self.negative_sample_size+1)
      
        if self.mode == 'hr_t':
            sample_h = torch.LongTensor([head]).expand(self.negative_sample_size+1)
            n_list = [[tail]]
        else:
            sample_t = torch.LongTensor([tail]).expand(self.negative_sample_size+1)
            n_list = [[head]]

        if self.mode == 'h_rt': # 替换尾实体
            negative_sample =(np.random.randint(0,len(self.t2disjoint[tail]), size=self.negative_sample_size))
            negative_sample_list =  self.t2disjoint[tail][negative_sample]
        else: # 如果小于则替换头实体
            negative_sample =(np.random.randint(0,len(self.h2disjoint[head]), size=self.negative_sample_size))
            negative_sample_list =  self.h2disjoint[head][negative_sample]

        n_list.append(negative_sample_list)
        negative_sample = np.concatenate(n_list)
        if self.mode == 'hr_t':
            sample_t =  torch.LongTensor(negative_sample)
        else:
            sample_h = torch.LongTensor(negative_sample)
        shuffle_idx = torch.randperm(sample_h.nelement())
        return sample_h[shuffle_idx],sample_r[shuffle_idx], sample_t[shuffle_idx], self.true_value[shuffle_idx],self.mode     
    
    @staticmethod
    def collate_fn(data):
        h = torch.stack([_[0] for _ in data], dim=0)
        r = torch.stack([_[1] for _ in data], dim=0)
        t = torch.stack([_[2] for _ in data], dim=0)
        v = torch.stack([_[3] for _ in data], dim=0)
        mode = data[0][4]
        return h,r,t,v,mode
    
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


def train_disjoint_softplus(train_iterator,model,cuda):
    model.train()
    positive_sample,negative_sample , mode = next(train_iterator)
    if cuda:
        positive_sample = positive_sample.cuda()
        negative_sample = negative_sample.cuda()
   
    h = positive_sample[:,0]
    r = positive_sample[:,1]
    t = positive_sample[:,2]
    positive_score = model(h,r,t)
    if mode == "hr_t":
        negative_score = model(h,r, negative_sample, mode=mode)
    else:
        negative_score = model(negative_sample,r, t, mode=mode)
    p_loss = model.criterion(-positive_score)
    n_loss = model.criterion(1.2*negative_score)
    p_loss = p_loss.unsqueeze(1)
    loss = torch.cat([p_loss, n_loss],dim=-1)
    loss = loss.sum(dim=1)
    loss = loss.mean()

    log = {
        'ons_disjoint_loss': loss.item(),
        # level+'_regu': regularization.item()
    }
    return log, loss

def train_disjoint_softplus_convkb(train_iterator,model,cuda):
    model.train()
    h,r,t,v,mode = next(train_iterator)
    if cuda:
       h = h.cuda()
       r = r.cuda()
       t = t.cuda()
       v = v.cuda()
   
    score, regu = model(h,r,t)
    loss = model.loss(score, v)

    log = {
        'ons_disjoint_loss': loss.item(),
        # level+'_regu': regularization.item()
    }
    return log, loss