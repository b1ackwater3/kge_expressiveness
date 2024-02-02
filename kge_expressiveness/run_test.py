#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import logging
import os
import random

import numpy as np
import torch

from trainer import OneToNTrainer,ModelTester

from util.data_process import DataProcesser as Data
from util.tools import logset

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR

from util.model_util import Trainer

from util.dataloader import NagativeSampleDataset
from util.dataloader import BidirectionalOneShotIterator
from torch.optim.lr_scheduler import StepLR

from core.TransE import TransE
from loss import NSSAL

from config.config import parse_args,parse_args_test

def override_config(args):
    '''
    Override model and data configuration
    '''
    with open(os.path.join(args.init_checkpoint, 'config.json'), 'r') as fjson:
        argparse_dict = json.load(fjson)
    
    args.countries = argparse_dict['countries']
    if args.data_path is None:
        args.data_path = argparse_dict['data_path']
    args.model = argparse_dict['model']
    args.double_entity_embedding = argparse_dict['double_entity_embedding']
    args.double_relation_embedding = argparse_dict['double_relation_embedding']
    args.hidden_dim = argparse_dict['hidden_dim']
    args.test_batch_size = argparse_dict['test_batch_size']
    
def save_dic(args, entity2id, relation2id):
    with open(os.path.join(args.data_path, 'entities_extend.dict'),'w', encoding='utf8') as fout:
        for key in entity2id.keys():
            fout.write("%s\t%d\n" % (key,entity2id[key]))
    with open(os.path.join(args.data_path, 'relationa_extend.dict'),'w', encoding='utf8') as fout:
        for key in relation2id.keys():
            fout.write("%s\t%d\n" % (key,relation2id[key])) 

def read_dataset(args):
    data = Data(args,idDict=True,reverse=True)
    classTest_data = None
    if args.do_test_class:
        #tod: 分类测试的读取时，同样需要all_true_triples, nentity,nrelation 等等数据，所以最好是也能够包括
        classTest_data = Data(args, "ClassTest")
    return data, classTest_data

def nagativeSample_dataset(train_triples,nentity,nrelation, args):
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

def main(args):
    if args.data_path is None:
        raise ValueError('one of init_checkpoint/data_path must be choosed.')
    if args.do_train and args.save_path is None:
        raise ValueError('Where do you want to save your trained model?')
    if args.save_path and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    data,classTest_data = read_dataset(args)

    args.nentity = data.nentity
    args.nrelation= data.nrelation
    logset.set_logger(args)

    # logging information about dataset and model
    logging.info('Model: %s' % args.model)
    logging.info('Data Path: %s' % args.data_path)
    logging.info('#entity: %d' % data.nentity)
    logging.info('#relation: %d' % data.nrelation)
    logging.info('#train: %d' % len(data.train))
    logging.info('#valid: %d' % len(data.valid))
    logging.info('#test: %d' % len(data.test))
    
    kge_model = TransE(
        n_entity=data.nentity,
        n_relation=data.nrelation,
        dim=args.hidden_dim
    )

    logging.info('Model Parameter Configuration:')
    for name, param in kge_model.named_parameters():
        logging.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))
    
    train_iterator = nagativeSample_dataset(data.train, data.nentity, data.nrelation, args)
    current_learning_rate = args.learning_rate
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, kge_model.parameters()), 
        lr=current_learning_rate
    )

    lr_scheduler = StepLR(optimizer, args.warm_up_steps, args.decay)

    loss_function = NSSAL(args.gamma,True,args.adversarial_temperature)
    trainer = Trainer(
        data=data,
        train_iterator=train_iterator,
        model=kge_model,
        optimizer=optimizer,
        loss_function=loss_function,
        args=args,
        logging=logging,
        lr_scheduler=lr_scheduler
        )
    
    trainer._init_model()
    trainer.logging_traing_info()
    trainer.train_model_()
    # metrics = Trainer.test_step(kge_model, data.test, data.all_true_triples, args)
    # logset.log_metrics('Test',0, metrics)

if __name__ == '__main__':
    main(parse_args())
