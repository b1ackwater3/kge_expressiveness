# 该文件要能够通过参数来配置所有项目

#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from torch.utils.tensorboard import SummaryWriter  
import argparse
from ast import arg
import json
import logging
import os
import random
from util.dataloader import TestDataset
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR,MultiStepLR

from util.model_util import Trainer,ModelTester

from util.data_process import DataProcesser as Data
from util.data_process import DataProcesser as DP

from util.data_process import Relation2Patter

from util.tools import logset

from core import *
import core
from loss import NSSAL,MRL


from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
from util.model_util import Trainer

from util.dataloader import NagativeSampleDataset,OneToNDataset
from util.dataloader import BidirectionalOneShotIterator,OneShotIterator

from config.config import parse_args

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


def read_dataset(args):
    data = Data(args.data_path, reverse=args.data_reverse)
    classTest_data = None
    if args.do_test_class:
        classTest_data = Data(args.data_path, dataType="ClassTest")
    
    return data, classTest_data

def bi_nagativeSampleDataset(train_triples,nentity,nrelation, args):
    train_dataloader_head = DataLoader(
        NagativeSampleDataset(train_triples, nentity, nrelation, args.negative_sample_size, 'h_rt'), 
        batch_size=args.batch_size,
        shuffle=True, 
        num_workers=max(1, args.cpu_num//2),
        collate_fn=NagativeSampleDataset.collate_fn
    )
    train_dataloader_tail = DataLoader(
        NagativeSampleDataset(train_triples, nentity, nrelation, args.negative_sample_size, 'hr_t'), 
        batch_size=args.batch_size,
        shuffle=True, 
        num_workers=max(1, args.cpu_num//2),
        collate_fn=NagativeSampleDataset.collate_fn
    )   
    train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)
    return train_iterator

def OneToN_data_iterator(train_triples,nentity,nrelation, args):
    train_dataloader = DataLoader(
        OneToNDataset(train_triples,nentity,nrelation),
        batch_size=args.batch_size,
        shuffle=True, 
        num_workers=max(1, args.cpu_num//2),
    )
    return OneShotIterator(train_dataloader)


def buildModel(args,data):
    model = None
    name_to_model = {
        "TransE":TransE,
        "TransH":TransH,
        "PairRE":PairRE,
        "RotatE":RotatE,
        "RotPro":RotPro,
    }
    model_class = name_to_model[args.model]

    if args.model in ["TransE","DistMult","HoLE","ComplEx","PairRE","RotatE","TorusE"]:
        model =  model_class(
            n_entity=data.nentity,
            n_relation=data.nrelation,
            dim=args.hidden_dim,
            gamma = args.gamma
        )
    elif args.model in  ["TransR"]:
        model = model_class(
            n_entity=data.nentity,
            n_relation=data.nrelation,
            entity_dim=args.hidden_dim,
            relation_dim=args.relation_dim
        )
       
    elif args.model in ["TuckER"]:
        model = model_class(
            n_entity=data.nentity,
            n_relation=data.nrelation,
            entity_dim=args.hidden_dim,
            relation_dim = args.relation_dim,
            dropout1 = args.dropout1,
            dropout2 = args.dropout2,
            dropout3 = args.dropout3
        ) 
    elif args.model in ["RotPro"]:
        model = model_class(
            n_entity=data.nentity,
            n_relation=data.nrelation,
            dim=args.hidden_dim,
            gamma = args.gamma
        ) 
    elif args.model in ['TransH']:
        model =  model_class(
            n_entity=data.nentity,
            n_relation=data.nrelation,
            dim=args.hidden_dim,
            gamma = args.gamma,
            dropout = args.transh_dropout
        )
    else:
        raise ValueError("Unknown model")

    return model


def override_config(args):
    '''
    Override model and data configuration
    '''
    with open(os.path.join(args.init_checkpoint, 'config.json'), 'r') as fjson:
        argparse_dict = json.load(fjson)
    if args.data_path is None:
        args.data_path = argparse_dict['data_path']
    args.model = argparse_dict['model']
    args.lr_step = argparse_dict['lr_step']
    args.relation_dim = argparse_dict['relation_dim']
    args.double_entity_embedding = argparse_dict['double_entity_embedding']
    args.double_relation_embedding = argparse_dict['double_relation_embedding']
    args.hidden_dim = argparse_dict['hidden_dim']
    args.loss_function = argparse_dict['loss_function']
    args.optimizer = argparse_dict['optimizer']
    args.train_type = argparse_dict['train_type']
    args.test_batch_size = argparse_dict['test_batch_size']

def buildLoss(name,args):
    name_to_loss={
        "NSSAL":NSSAL,
        "MRL": MRL,
        "BCE": torch.nn.BCELoss,
        "CrossEntropyLoss":torch.nn.CrossEntropyLoss
    }
    if name not in name_to_loss:
        raise ValueError("Sorry! Unknown Loss Name, you can implement it by yourself")
    if name in ["NSSAL","MRL"]:
        loss = name_to_loss[name](gamma=args.gamma)
    else:
        loss = name_to_loss[name]()
    return loss


def buildOptimizer(name):
    name_to_optimizer ={
        "Adam":torch.optim.Adam
    }
    return name_to_optimizer[name]

def main(args):
    
    if args.data_path is None:
        raise ValueError('one of init_checkpoint/data_path must be choosed.')
    if args.do_train and args.save_path is None:
        raise ValueError('Where do you want to save your trained model?')
    if args.save_path and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    writer = SummaryWriter(os.path.join(args.save_path,"log"))

    model_name = str(args.save_path).split("/")[-1]

    if not args.do_train:
        logset.set_logger(args.save_path,'temp.log')
    else:
        logset.set_logger(args.save_path)


    # if args.init_checkpoint:
    #     override_config(args)

    data, classTest_data = read_dataset(args)
    model = buildModel(args,data)
    if args.cuda:
        model =model.cuda()

    loss = buildLoss(args.loss_function, args)
    current_learning_rate = args.learning_rate
    optimizer_class = buildOptimizer(args.optimizer)
    optimizer = optimizer_class(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=current_learning_rate)
    
    args.nentity = data.nentity
    args.nrelation = data.nrelation
    if args.init_checkpoint:
        logging.info('Loading checkpoint %s...' % args.init_checkpoint)
        checkpoint = torch.load(os.path.join(args.init_checkpoint, 'checkpoint'))
        args.init_step = checkpoint['step']
        print(checkpoint['model_state_dict'].keys())
        model.load_state_dict(checkpoint['model_state_dict'],strict=True)

        if args.do_train:
            current_learning_rate = checkpoint['current_learning_rate']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        logging.info('Ramdomly Initializing %s Model...' % args.model)
        args.init_step = 0

    step = args.init_step

    logging.info('Model: %s' % args.model)
    logging.info('Data Path: %s' % args.data_path)
    logging.info('#entity: %d' % data.nentity)
    logging.info('#relation: %d' % data.nrelation)
    logging.info('#train: %d' % len(data.train))
    logging.info('#valid: %d' % len(data.valid))
    logging.info('#test: %d' % len(data.test))
    if args.cuda:
        logging.info('Device: %s' % ("CUDA"))
    else:
        logging.info('Device: %s' % ("CPU"))
    
    is_on_train = False
    if args.do_train:
        is_on_train = True
        lr_scheduler = StepLR(optimizer, args.lr_step, args.decay)
        # lr_scheduler = MultiStepLR(optimizer,milestones=[50000,120000], gamma=args.decay)
        print(args.lr_step)
        if args.train_type == '1toN':
            train_iterator = OneToN_data_iterator(data.train, data.nentity, data.nrelation, args)
        elif args.train_type == 'NagativeSample':
            train_iterator = bi_nagativeSampleDataset(data.train, data.nentity, data.nrelation,args)
        
        trainer = Trainer(
                data=data,
                train_iterator=train_iterator,
                model=model,
                optimizer=optimizer,
                loss_function=loss,
                args=args,
                lr_scheduler=lr_scheduler,
                logging=logging,
                train_type=args.train_type,
                tb_writer= writer,
                model_name=model_name
                )
        trainer.logging_traing_info()
        trainer.train_model_(classTest_data)
        step = args.max_steps
    
    if args.do_test:
        if(is_on_train):
            path = os.path.join(args.save_path,'Normal')
            logging.info('Loading checkpoint %s...' % path)
            checkpoint = torch.load(os.path.join(path, 'checkpoint'))
            args.init_step = checkpoint['step']
            model.load_state_dict(checkpoint['model_state_dict'])        
        metrics = ModelTester.test_step(model, data.test, data.all_true_triples, args,loss_function=loss)
        logset.log_metrics('Test',step, metrics)
    
    if args.do_test_class:

        for relation in ["symmetry",'transitive']:
            if len(classTest_data.class_test_data[relation][0]) > 10:
                logging.info('Test relation pattern %s' % (relation))
                if relation != 'transitive':
                    metrics,test_case,triple_embedding_list = ModelTester.class_test_step_print_case(model,classTest_data.class_test_data[relation],data.all_true_triples,relation,args)
                else:
                    metrics,test_case,triple_embedding_list = ModelTester.class_test_step_print_case(model,classTest_data.class_test_data[relation],classTest_data.all_transtive,relation,args)
                logset.log_metrics('Test',step, metrics)

if __name__ == '__main__':
    main(parse_args())
