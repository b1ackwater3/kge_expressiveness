from collections import defaultdict
from curses import init_pair

from pickle import FALSE
from tkinter.tix import Tree
from util.data_process import DataProcesser as DP
from util.data_process import DataTool
import os
from core.RotPro import RotPro
from core.TransE import TransE
from core.RotatE import RotatE
from core.PairRE import PairRE
import numpy as np

import torch
from torch.optim.lr_scheduler import StepLR,ReduceLROnPlateau,MultiStepLR
from util.dataloader import NagativeSampleDataset,BidirectionalOneShotIterator,OneShotIterator,NagativeSampleNewDataset
from loss import NSSAL,MRL
from util.tools import logset
from util.model_util import ModelUtil,ModelTester
from torch.utils.data import DataLoader
from util.dataloader import TestDataset
import logging
from torch.optim.lr_scheduler import StepLR

import argparse
import random




def logging_log(step, logs):
    metrics = {}
    for metric in logs[0].keys():
        metrics[metric] = sum([log[metric] for log in logs])/len(logs)
    logset.log_metrics('Training average', step, metrics)

def train_step_old(train_iterator,model,loss_function,cuda, args,disjoint=None,isMarry=None, isConnect=None):
    positive_sample,negative_sample, subsampling_weight, mode = next(train_iterator)
    if cuda:
        positive_sample = positive_sample.cuda()
        negative_sample = negative_sample.cuda()
        subsampling_weight = subsampling_weight.cuda()
   
    h = positive_sample[:,0]
    r = positive_sample[:,1]
    t = positive_sample[:,2]
    if mode =='hr_t':
        negative_score = model(h,r, negative_sample,mode)
    else:
        negative_score = model(negative_sample,r, t,mode)
    positive_score = model(h,r,t)
   
    loss = loss_function(positive_score, negative_score,subsampling_weight)
    # loss = loss_function(positive_score, negative_score)
    # loss +=model.caculate_constarin_sys()
    if True:
        reg = args.loss_weight * model.caculate_constarin_sys(h,t)
        loss_1 = loss  + reg
        log = {
            '_loss': loss.item(),
            'regul': reg.item()
        }
    else:
        loss_1 = loss
        log = {
            '_loss': loss.item(),
        }
    return log, loss_1
    
def test_step_f(model, test_triples, all_true_triples,nentity,nrelation,cuda=True, inverse=False, onType=None):
    '''
    Evaluate the model on test or valid datasets
    '''
    model.eval()
    test_dataloader_tail = DataLoader(
        TestDataset(
            test_triples, 
            all_true_triples, 
            nentity, 
            nrelation, 
            'hr_t'
        ), 
        batch_size=8,
        num_workers=1, 
        collate_fn=TestDataset.collate_fn
    )
    test_dataloader_head = DataLoader(
        TestDataset(
            test_triples, 
            all_true_triples, 
            nentity, 
            nrelation, 
            'h_rt'
        ), 
        batch_size=8,
        num_workers=1, 
        collate_fn=TestDataset.collate_fn
    )
    if not onType is None:
        if onType == 'head':
            test_dataset_list = [test_dataloader_head]
        else:
            test_dataset_list = [test_dataloader_tail]
    else:
        if not inverse:
            test_dataset_list = [test_dataloader_tail,test_dataloader_head]
        else:
            test_dataset_list = [test_dataloader_tail]
    logs = []
    step = 0
    total_steps = sum([len(dataset) for dataset in test_dataset_list])
    count = 0
    # print(total_steps)
    with torch.no_grad():
        for test_dataset in test_dataset_list:
            for positive_sample, negative_sample, filter_bias, mode in test_dataset:
                batch_size = positive_sample.size(0)
                if cuda:
                    positive_sample = positive_sample.cuda()
                    negative_sample = negative_sample.cuda()
                    filter_bias = filter_bias.cuda()
                    
                h = positive_sample[:,0]
                r = positive_sample[:,1]
                t = positive_sample[:,2] 
                if mode == 'hr_t':
                    negative_score = model(h,r, negative_sample,mode=mode)
                    positive_arg = t
                else:
                    negative_score = model(negative_sample,r,t,mode=mode)
                    positive_arg = h
                # 
                score = negative_score + filter_bias
                argsort = torch.argsort(score, dim = 1, descending=True)
                

                for i in range(batch_size):
                    count = count + 1
                    ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                    assert ranking.size(0) == 1
                    ranking = 1 + ranking.item()
                    logs.append({
                        'MRR': 1.0/ranking,
                        'MR': float(ranking),
                        'HITS@1': 1.0 if ranking <= 1 else 0.0,
                        'HITS@3': 1.0 if ranking <= 3 else 0.0,
                        'HITS@10': 1.0 if ranking <= 10 else 0.0,
                    })
                step += 1
    metrics = {}
    for metric in logs[0].keys():
        metrics[metric] = sum([log[metric] for log in logs])/len(logs)
    metrics["Test Count"] = count
    return metrics

def set_config(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )
    parser.add_argument('--cuda', action='store_true', help='use GPU')
    parser.add_argument('--train', action='store_true', help='train model')
    parser.add_argument('--test', action='store_true', help='test model')
    parser.add_argument('--valid', action='store_true', help='valid model')
    
    parser.add_argument('--max_step', type=int,default=200001, help='最大的训练step')
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--test_step", type=int, default=10000)
    parser.add_argument("--neg_size",type=int, default=256)
    parser.add_argument("--gamma", type=float, default=20)
    parser.add_argument("--adversial_temp", type=float, default=0.5)

    parser.add_argument("--dim", type=int, default=200)

    parser.add_argument("--lr", type=float)
    parser.add_argument("--decay", type=float)
    parser.add_argument("--warm_up_step", type=int, default=50000)

    parser.add_argument("--loss_function", type=str)

    # RotPro 约束参数配置
    parser.add_argument("--gamma_m",type=float,default=0.000001)
    parser.add_argument("--alpha",type=float,default=0.0005)
    parser.add_argument("--beta",type=float,default=1.5)
    parser.add_argument("--train_pr_prop",type=float,default=1)
    parser.add_argument("--loss_weight",type=float,default=1)

    # 选择数据集
    parser.add_argument("--level",type=str,default='ins')
    parser.add_argument("--data_inverse",action='store_true')
    

    return parser.parse_args(args)


def save_embedding(emb_map,file):
    for key in emb_map.keys():
        
        path = os.path.join(file,key)
        print((emb_map[key].detach().cpu().numpy().shape))
        print(path)
        np.save(
            path,emb_map[key].detach().cpu().numpy()
        )


def build_dataset(data_path,args):
    on = DP(os.path.join(data_path),idDict=True, reverse=args.data_inverse)
   
    on_train_t = DataLoader(NagativeSampleDataset(on.train, on.nentity, on.nrelation, n_size, 'hr_t'),
            batch_size=batch_size,
            shuffle=True, 
            num_workers=max(1, 4//2),
            collate_fn=NagativeSampleDataset.collate_fn
    )
    if(not args.data_inverse):
        on_train_h = DataLoader(NagativeSampleDataset(on.train, on.nentity, on.nrelation, n_size, 'h_rt'),
            batch_size=batch_size,
            shuffle=True, 
            num_workers=max(1, 4//2),
            collate_fn=NagativeSampleDataset.collate_fn
        )
        on_train_iterator = BidirectionalOneShotIterator(on_train_h, on_train_t)
    else:
        on_train_iterator = OneShotIterator(on_train_t)
        
    on_valid = DataLoader(NagativeSampleDataset(on.valid, on.nentity, on.nrelation, 2, 'hr_t'),
            batch_size=batch_size,
            shuffle=True, 
            num_workers=max(1, 4//2),
            collate_fn=NagativeSampleDataset.collate_fn
    )
    on_test = DataLoader(NagativeSampleDataset(on.test, on.nentity, on.nrelation, 2, 'hr_t'),
            batch_size=batch_size,
            shuffle=True, 
            num_workers=max(1, 4//2),
            collate_fn=NagativeSampleDataset.collate_fn
    )
    
    on_valid_ite = OneShotIterator(on_valid)
    on_test_ite = OneShotIterator(on_test)
    on_ite = {
        "train": on_train_iterator,
        "valid": on_valid_ite,
        "test": on_test_ite
    }
    return on,on_ite

def getModel(model_name):
    model_dic={
        'TransE':TransE,
        "RotatE":RotatE,
        'RotPro':RotPro,
    }
    return model_dic[model_name]


def test_mapping(step,dataset,cuda):
    data_path ="/home/skl/yl/FB15k/mapping"
    test_file = ['121.txt','12n.txt','n21.txt','n2n.txt']
    for file_name in test_file:
        file = os.path.join(data_path,file_name)
        triples = DP.read_triples(file,entity2id=dataset.entity2id,relation2id=dataset.relation2id)
        logging.info('Test at patter: %s' % file_name[:-4])
        logging.info('predicate Head : %s' % file_name[:-4])
        metrics = test_step_f(model, triples, dataset.all_true_triples,dataset.nentity,dataset.nrelation,cuda,onType='head')
        logset.log_metrics('Valid ',step, metrics)
        logging.info('predicate Tail : %s' % file_name[:-4])
        metrics = test_step_f(model, triples, dataset.all_true_triples,dataset.nentity,dataset.nrelation,cuda,onType='tail')
        logset.log_metrics('Valid ',step, metrics)


if __name__=="__main__":
    # 读取4个数据集
    args = set_config()

    cuda = args.cuda
    init_step  = 0
    save_steps = 10000

    root_path = os.path.join("/home/skl/yl/models/",args.save_path)
    args.save_path = root_path

    init_path = args.init
    max_step   = args.max_step
    batch_size = args.batch_size
    test_step = args.test_step
    dim = args.dim

    lr = args.lr
    decay = args.decay
    warm_up_steps = args.warm_up_step

    g_ons = args.gamma
    # n_size = 256
    n_size = 400
    log_steps = 1000

    data_path ="/home/skl/yl/data/YAGO-sys"


    if not os.path.exists(root_path):
        os.makedirs(root_path)
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    if args.train:
        logset.set_logger(root_path,"init_train.log")
    else:
        logset.set_logger(root_path,'test.log')
    
 

    # 读取数据集：
    dataset, ite = build_dataset(data_path,args)

    logging.info('Model: %s' % args.model)
    logging.info('On nentity: %s' % dataset.nentity)
    logging.info('On nrelatidataset. %s' % dataset.nrelation)
    logging.info('max step: %s' % max_step)
    logging.info('gamma: %s' % g_ons)
    logging.info('lr: %s' % lr)


    model = PairRE(dataset.nentity, dataset.nrelation, dim,gamma=g_ons)
    if cuda:
        model = model.cuda()
    
    loss_function_on = NSSAL(g_ons,True,adversarial_temperature=args.adversial_temp)
  
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=lr
    )

    # 0-isMarried 1-isConn
    isMarry = set()
    isConn = set()
    for h,r,t in dataset.train:
        if r == 0:
            isMarry.add(h)
            isMarry.add(t)
        else:
            isConn.add(h)
            isConn.add(t)

    isMarry = list(isMarry)
    isConn= list(isConn)
   

    # 如果有保存模型则，读取模型,进行测试
    if init_path != None:
        logging.info('init: %s' % init_path)
        checkpoint = torch.load(os.path.join(init_path, 'checkpoint'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        init_step = checkpoint['step']
        save_embedding(model.save_model_embedding(),root_path)


    # 设置学习率更新策略
    # lr_scheduler = StepLR(optimizer, warm_up_steps, decay,verbose=False)
    # lr_scheduler = ReduceLROnPlateau(optimizer,mode='min',factor=0.1)
    lr_scheduler = MultiStepLR(optimizer,milestones=[50000], gamma=decay)
    # lr_scheduler = MultiStepLR(optimizer,milestones=[30000,70000,110000,160000], gamma=decay)
    logsB = []
    stepW = 0
    bestModel = {
        "MRR":0,
        "MR":100000,
        "HITS@1":0,
        "HITS@3":0,
        "HITS@10":0
    }

    if args.train :
        for step in range(init_step, max_step):
            stepW = (step//10000) % 2
            optimizer.zero_grad(set_to_none=True)
            log2,loss_on = train_step_old(ite['train'],model,loss_function_on,cuda,args,isMarry=isMarry,isConnect=isConn)
            loss_on.backward()
            optimizer.step()
            lr_scheduler.step()
            logsB.append(log2)

            if step % log_steps == 0 :
                logging_log(step,logsB)
                logsB = []

            if step % test_step == 0  and step != 0:
                save_variable_list = {"lr":lr_scheduler.get_last_lr(),"step":step,
                    "gamma":g_ons, "dim":dim,
                }

                logging.info('Valid at step: %d' % step)
                metrics = test_step_f(model, dataset.valid, dataset.all_true_triples,dataset.nentity,dataset.nrelation,cuda,args.data_inverse)
                logset.log_metrics('Valid ',step, metrics)

        logging.info('Test at step: %d' % step)
        metrics = test_step_f(model, dataset.test, dataset.all_true_triples,dataset.nentity,dataset.nrelation,cuda,args.data_inverse)
        logset.log_metrics('Test ',step, metrics)
                
        save_variable_list = {"lr":lr_scheduler.get_last_lr(),"step":step,
           "gamma":g_ons, "dim":dim,
        }
        ModelUtil.save_model(model,optimizer,save_variable_list=save_variable_list,path=root_path,args=args)

        isMarry = []
        isConnect = []
        for h,r,t in dataset.test:
            if r == 0:
                isMarry.append((h,r,t))
            else:
                isConnect.append((h,r,t))

                
        logging.info('Test relation(IsMarriedTo) at step: %d' % step)
        metrics = test_step_f(model, isMarry, dataset.all_true_triples,dataset.nentity,dataset.nrelation,cuda,args.data_inverse)
        logset.log_metrics('Test ',step, metrics)
        logging.info('Test relation(isConnectedTo) at step: %d' % step)
        metrics = test_step_f(model, isConnect, dataset.all_true_triples,dataset.nentity,dataset.nrelation,cuda,args.data_inverse)
        logset.log_metrics('Test ',step, metrics)       
                
        classTest_data = DP(data_path, dataType="ClassTest")

        for relation in ['symmetry']:
            logging.info('Test relation pattern %s' % (relation))
            args.nrelation = dataset.nrelation
            args.nentity = dataset.nentity
            metrics = ModelTester.class_test_step(model,classTest_data.class_test_data[relation],dataset.all_true_triples,relation,args)
            logset.log_metrics('Test',step, metrics) 
        

    step = max_step       
    if args.test:

        if args.level == '15k':
            data_path ="/home/skl/yl/FB15k/mapping"
            test_file = ['121.txt','12n.txt','n21.txt','n2n.txt']
            for file_name in test_file:
                file = os.path.join(data_path,file_name)
                triples = DP.read_triples(file,entity2id=dataset.entity2id,relation2id=dataset.relation2id)
                logging.info('Test at patter: %s' % file_name[:-4])
                logging.info('predicate Head : %s' % file_name[:-4])
                metrics = test_step_f(model, triples, dataset.all_true_triples,dataset.nentity,dataset.nrelation,cuda,onType='head')
                logset.log_metrics('Valid ',step, metrics)
                logging.info('predicate Tail : %s' % file_name[:-4])
                metrics = test_step_f(model, triples, dataset.all_true_triples,dataset.nentity,dataset.nrelation,cuda,onType='tail')
                logset.log_metrics('Valid ',step, metrics)
        else:
            classTest_data = DP(data_path, dataType="ClassTest")

            for relation in ['symmetry']:
                logging.info('Test relation pattern %s' % (relation))
                metrics = ModelTester.class_test_step(model,classTest_data.class_test_data[relation],dataset.all_true_triples,relation,args)
                logset.log_metrics('Test',step, metrics)

        