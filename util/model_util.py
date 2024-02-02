
import torch
import numpy as np
import os
import json

import torch.nn.functional as F
from util.dataloader import TestDataset
import logging
from util.dataloader import MulTestDataset
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader
from util.tools import logset
import math
def ndcg_at_k(idx):
    idcg_k = 0
    dcg_k = 0
    n_k = 1
    for i in range(n_k):
        idcg_k += 1 / math.log(i + 2, 2)
    dcg_k += 1 / math.log(idx + 2, 2)
    return float(dcg_k / idcg_k)  

class ModelUtil(object):
    @staticmethod
    def save_best_model(metrics,best_metrics,model, optimizer,save_variable_list,args, path=None):
        for key in metrics.keys():
            if metrics[key] > best_metrics[key] and key != "MR": 
                best_metrics[key] = metrics[key]
                logging.info('Save models for best %s until now' % key)
                ModelUtil.save_model(model, optimizer, save_variable_list, args,type=key,path=path)
        if "MR" in metrics.keys():
            if metrics['MR'] < best_metrics['MR']:
                best_metrics['MR'] = metrics['MR']
                logging.info('Save models for best MR until now')
                ModelUtil.save_model(model, optimizer, save_variable_list, args,type='MR',path=path)

    @staticmethod
    def init_model(model, optimizer,path):
        checkpoint = torch.load(os.path.join(path, 'checkpoint'))
        init_step = checkpoint['step']
        model.load_state_dict(checkpoint['model_state_dict'])
        current_learning_rate = checkpoint['current_learning_rate']
        warm_up_steps = checkpoint['warm_up_steps']
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    @staticmethod
    def save_model(model, optimizer, save_variable_list=None, args=None, type=None, path=None):
        '''
        Save the parameters of the model and the optimizer,
        as well as some other variables such as step and learning_rate
        '''
        if path != None:
            root_path = path
        else:
            root_path = args.save_path

        if type is not None:
            root_path = os.path.join(root_path,type)
        if  not os.path.exists(root_path):
            os.makedirs(root_path)
            
        if args != None:
            argparse_dict = vars(args)
            with open(os.path.join(root_path, 'config.json'), 'w') as fjson:
                json.dump(argparse_dict, fjson)
        torch.save({
            **save_variable_list,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()},
            os.path.join(root_path, 'checkpoint')
        )

class Trainer(object):
    # train_type:  NagativeSample and 1toN
    def __init__(self, data, train_iterator, model, optimizer, loss_function, args, lr_scheduler=None,logging=None,train_type='NagativeSample',tb_writer=None,model_name=None):
        self.args = args
        self.train_type = train_type
        self.model = model
        self.optimizer = optimizer
        # 
        self.model_name = model_name
        self.step = 0
        self.step = args.init_step
        self.data = data
        self.max_step = args.max_steps
        self.train_iterator = train_iterator
        self.lr_scheduler = lr_scheduler
        self.loss_function = loss_function
        self.logging = logging

        # self.best_metrics = {'MRR':0.0, 'MR':1000000000, 'HITS@1':0.0,'HITS@3':0.0,'HITS@10':0.0}
        self.best_metrics = {'Normal':0.0, 'SysClass':0.0, 'TransClass':0.0}


        self.tb_write = tb_writer

    def _init_model(self):
        if self.args.init_checkpoint:
            checkpoint = torch.load(os.path.join(self.args.init_checkpoint, 'checkpoint'))
            self.init_step = checkpoint['step']
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.current_learning_rate = checkpoint['current_learning_rate']
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            self.init_step = 0

    def logging_model_info(self):
        pass

    def logging_traing_info(self):
        logging = self.logging
        logging.info('Start Training...')
        logging.info('init_step = %d' % self.step)
        logging.info('max_step = %d' % self.max_step)
        logging.info('batch_size = %d' % self.args.batch_size)
        logging.info('negative_adversarial_sampling = %d' % self.args.negative_adversarial_sampling)
        logging.info('hidden_dim = %d' % self.args.hidden_dim)
        logging.info('gamma = %f' % self.args.gamma)
        logging.info('negative_adversarial_sampling = %s' % str(self.args.negative_adversarial_sampling))
        if self.args.negative_adversarial_sampling:
            logging.info('adversarial_temperature = %f' % self.args.adversarial_temperature)

    def train_model_(self,classTest_data=None):
        Trainer.train_model(data=self.data,train_iterator=self.train_iterator,
                    model=self.model,optimizer=self.optimizer,loss_function=self.loss_function,
                    max_step=self.max_step, init_step=self.step,
                    args=self.args,best_metrics=self.best_metrics,
                    lr_scheduler=self.lr_scheduler,train_type=self.train_type,tb_writer=self.tb_write,model_name=self.model_name,classTest_data=classTest_data)

    # 数据集产生 ground truth：损失基于ground truth 和 预测结果计算
    @staticmethod
    def train_step_1(model, optimizer, train_iterator,loss_function,args,tb_writer=None):
        model.train()
        optimizer.zero_grad()
        h_and_rs, ground_truth, mode = next(train_iterator)

        if args.cuda:
            h_and_rs = h_and_rs.cuda()
            ground_truth = ground_truth.cuda()
           
        ground_truth.detach()   
        ground_truth = ground_truth 
        if args.label_smoothing != 0.0:
            ground_truth = ((1.0-args.label_smoothing)*ground_truth) + (1.0/ground_truth.size(1))  
            
        if mode[0] == 'hr_all':
            h = h_and_rs[:,0]
            r = h_and_rs[:,1]
            pre_score = model(h,r, None, mode=mode[0])
        else:
            t = h_and_rs[:,0]
            r = h_and_rs[:,1]
            pre_score = model(None,r, t, mode=mode[0])
        pre_score = torch.sigmoid(pre_score)
        loss = loss_function(pre_score,ground_truth)
        loss.backward()
        optimizer.step()

        log = {
            'loss': loss.item()
        }
        return log

    # 训练负采样的数据集: 数据集产生正样本和负样本, 损失基于正负样本的计算
    @staticmethod
    def train_step(model, optimizer, train_iterator,loss_function, args,tb_writer=None):
        '''
        A single train step. Apply back-propation and return the loss
        '''

        model.train()
        optimizer.zero_grad()
        positive_sample,negative_sample, subsampling_weight, mode = next(train_iterator)

        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()
       
        h = positive_sample[:,0]
        r = positive_sample[:,1]
        t = positive_sample[:,2]
        if mode == 'hr_t':
            negative_score = model(h,r, negative_sample, mode=mode)
        else:
            negative_score = model(negative_sample,r,t, mode=mode)

        positive_score= model(h,r,t)
        if args.loss_function == 'MRL':
            loss = loss_function(positive_score, negative_score)
        else:
            loss = loss_function(positive_score, negative_score,subsampling_weight)

        regularization = 0
        if args.regularization != 0.0:
            regularization = args.regularization * (
                model.entity_embedding.weight.data.norm(p = 3)**3 + 
                model.relation_embedding.weight.data.norm(p = 3).norm(p = 3)**3
            )
            
        if args.model in ['RotPro'] and args.do_rel_constrain:
            regularization =  model.caculate_constarin(args.gamma_m,args.beta,args.alpha)
            # h1,h2 =  model.caculate_constarin(args.gamma_m,args.beta,args.alpha)
            # pharse,h =  model.caculate_constarin_phare(args.rgamma_m,args.rbeta,args.ralpha,gamma=args.rel_gam,leng_min=args.min_len,constype=args.constype)
            # h1.remove()
            # h2.remove()
            # else:
                # pharse=  model.caculate_constarin_phare(args.rgamma_m,args.rbeta,args.ralpha,gamma=args.rel_gam,leng_min=args.min_len,constype=args.constype)
            # regularization += pharse
            # regularization = 0

        if args.model in ['RotatE','TransH'] and args.do_rel_constrain:
            if args.constype == 'set':
                regularization,h =  model.caculate_constarin(args.gamma_m,args.beta,args.alpha,gamma=args.rel_gam,leng_min=args.min_len,constype=args.constype)
                h.remove()
            else:
                regularization, rel_log=  model.caculate_constarin(args.gamma_m,args.beta,args.alpha,gamma=args.rel_gam,leng_min=args.min_len,constype=args.constype)
        if args.model in ['PairRE'] and args.do_rel_constrain:
            if args.constype == 'set':
                regularization,h,t=  model.caculate_constarin(args.gamma_m,args.beta,args.alpha,gamma=args.rel_gam,leng_min=args.min_len,constype=args.constype)
                h.remove()
                t.remove()
            else:
                regularization=  model.caculate_constarin(args.gamma_m,args.beta,args.alpha,gamma=args.rel_gam,leng_min=args.min_len,constype=args.constype)

        loss += regularization

        loss.backward()
        optimizer.step()
        log = {
            'regularization':regularization,
            'loss': loss.item(),
            "lo": rel_log["lo"],
            "lz": rel_log["lz"]
        }
        return log

    @staticmethod
    def train_model(data, train_iterator, model, optimizer,loss_function, max_step, init_step,args,best_metrics,lr_scheduler=None, train_type="NagativeSample",tb_writer=None,model_name=None,classTest_data=None):
        training_logs = []

        if train_type == 'NagativeSample':
            TRAIN_STEP = Trainer.train_step
        else:
            TRAIN_STEP = Trainer.train_step_1

        sym_r_tensor = torch.LongTensor(list(classTest_data.symmetry_relation))
        tran_r_tensor = torch.LongTensor(list(classTest_data.transitive_relation))

        sym_r_tensor = sym_r_tensor.cuda()
        tran_r_tensor = tran_r_tensor.cuda()

        for step in range(init_step, max_step):
            log = TRAIN_STEP(model, optimizer,train_iterator,loss_function,args,tb_writer)

            loss, rel_log = model.caculate_constarin_sub(args.gamma_m,args.beta,args.alpha,gamma=args.rel_gam,constype=args.constype,rel_index=sym_r_tensor)
            log["sym_lo"] = rel_log["lo"]
            log["sym_lz"] = rel_log["lz"]

            loss, rel_log = model.caculate_constarin_sub(args.gamma_m,args.beta,args.alpha,gamma=args.rel_gam,constype=args.constype,rel_index=tran_r_tensor)
            log["trans_lo"] = rel_log["lo"]
            log["trans_lz"] = rel_log["lz"]

            if tb_writer:
                for key in log.keys():
                    tb_writer.add_scalar(key, log[key], step)
            training_logs.append(log)
            if lr_scheduler != None:
                lr_scheduler.step()
            
            if step % args.save_checkpoint_steps == 0:
                save_variable_list = {
                    'step': step, 
                    'current_learning_rate': optimizer.state_dict()['param_groups'][0]['lr'],
                }
                ModelUtil.save_model(model,optimizer,save_variable_list,args)
            if step % 100 == 0:
                metrics = {}
                for metric in training_logs[0].keys():
                    metrics[metric] = sum([log[metric] for log in training_logs])/len(training_logs)
                if tb_writer:
                    for key in metrics.keys():
                        tb_writer.add_scalar(key, metrics[key], step)
                logset.log_metrics('Training average', step, metrics)
                training_logs = []
            classTest_data
            if  step % 10000 == 0:
                save_variable_list = {
                    'step': step, 
                    'current_learning_rate': optimizer.state_dict()['param_groups'][0]['lr'],
                }
                new_metric = {}
                metrics = ModelTester.test_step(model, data.valid, data.all_true_triples, args)
                new_metric['Normal'] = metrics['HITS@1'] +  metrics['HITS@3'] + metrics['HITS@10']
                logset.log_metrics('Valid', step, metrics)
                # if classTest_data != None:
                #     logging.info('Test relation pattern symmetry')
                #     relation = 'symmetry'
                #     sys_metrics,test_case,triple_embedding_list = ModelTester.class_test_step_print_case(model,classTest_data.class_test_data[relation],data.all_true_triples,relation,args)
                #     logset.log_metrics('valid',step, sys_metrics)
                #     new_metric['SysClass'] = sys_metrics['HITS@1'] +  sys_metrics['HITS@3'] + sys_metrics['HITS@10']
                    
                #     # if len(classTest_data.class_test_data['transitive'][0]) > 10:
                #     #     logging.info('Test relation pattern transitivety')
                #     #     relation = 'transitive'
                #     #     transe_metrics,test_case,triple_embedding_list = ModelTester.class_test_step_print_case(model,classTest_data.class_test_data[relation],data.all_true_triples,relation,args)
                #     #     logset.log_metrics('valid',step, transe_metrics)
                #     #     new_metric['TransClass'] = 4*transe_metrics['HITS@1'] +  2*transe_metrics['HITS@3'] + transe_metrics['HITS@10']
                ModelUtil.save_best_model(new_metric,best_metrics, model,optimizer,save_variable_list,args)
        save_variable_list = {
            'step': max_step, 
            'current_learning_rate': optimizer.state_dict()['param_groups'][0]['lr'],
            # todo 其他需要保存的参数
        }
        ModelUtil.save_model(model,optimizer,save_variable_list,args)

class ModelTester(object):
    def __init__(self,model):
        self.model = model
    
    @staticmethod
    def class_test(model, tiple_list,all_true_triples,relation_types,args):
        for i in range(len(relation_types)):
            logging.info("Begin test relation %s " % relation_types[i])
            args.test_relation=relation_types[i]
            metrics,case_list =  ModelTester.class_test_step_print_case(model, tiple_list[i], all_true_triples, args,test_relation=relation_types[i]) 
            logset.log_metrics('Test', 0, metrics)
        return case_list

    @staticmethod
    def class_test_step(model, test_triples, all_true_triples, relation, args):
        '''
        Evaluate the model on test or valid datasets
        '''
        model.eval()
        size_triples = len(test_triples)
        # get some to test process
        # new_triples = []
        # for a in test_triples:
        #     new_triples.append(a[0:50])
        # test_triples= new_triples
        print(size_triples)

        test_dataloader_head= DataLoader(
                MulTestDataset(
                    test_triples, 
                    all_true_triples, 
                    args.nentity, 
                    args.nrelation, 
                    'h_rt'
                ), 
                batch_size=4,
                num_workers=1
            )
        test_dataloader_tail = DataLoader(
                MulTestDataset(
                    test_triples, 
                    all_true_triples, 
                    args.nentity, 
                    args.nrelation, 
                    'hr_t'
                ), 
                batch_size=4,
                num_workers=1
            )
        test_dataset_list = [test_dataloader_head, test_dataloader_tail]
        logs = []
        step = 0
        total_steps = sum([len(dataset) for dataset in test_dataset_list])
        test_num = 0
        step_hit10 = 0
        step_hit1 = 0
        step_hit3 = 0
        hit_false = 50
        print("Test Total", total_steps)
        with torch.no_grad():
            for test_dataset in test_dataset_list:
                for positive_samples, negative_samples, filter_biass, mode in test_dataset:
                    step +=1
                    if step % 100 == 0:
                        logging.info("ClassTest %d / %d" % (step, total_steps))
                    if args.cuda:
                        positive_samples = positive_samples.cuda()
                        negative_samples = negative_samples.cuda()
                        filter_biass = filter_biass.cuda()
                    positive_samples = list(positive_samples.split(1,1))
                    negative_samples = list(negative_samples.split(1,1))
                    filter_biass = list(filter_biass.split(1,1))
                    ranks=[[] for i in range(size_triples)]
                    mode = mode[0]
                    for triple_sort in range(size_triples):
                        positive_sample = torch.squeeze(positive_samples[triple_sort],dim=1) 
                        negative_sample = torch.squeeze(negative_samples[triple_sort],dim=1)
                        filter_bias = torch.squeeze(filter_biass[triple_sort],dim=1)
                        batch_size = positive_sample.size(0)
                        h = positive_sample[:,0]
                        r = positive_sample[:,1]
                        t = positive_sample[:,2]
                        if mode == 'hr_t':
                            score = model(h,r, negative_sample, mode=mode)
                        else:
                            score = model(negative_sample,r,t, mode=mode)
                        score += filter_bias
                        argsort = torch.argsort(score, dim = 1, descending=True)
                        if mode == 'h_rt':
                            positive_arg = positive_sample[:, 0]
                        elif mode == 'hr_t':
                            positive_arg = positive_sample[:, 2]
                        else:
                            raise ValueError('mode %s not supported' % mode)
                        for i in range(batch_size):
                            ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                            assert ranking.size(0) == 1
                            ranking = 1 + ranking.item()    
                            ranks[triple_sort].append(ranking)

                    for ks in range(len(ranks[0])):
                        ranking = None
                        is_hit10 = 0.0
                        is_hit3 = 0.0
                        is_hit1 = 0.0
                        test_num += 1
                        if relation == 'symmetry' or relation == 'inverse':
                            ranking = ranks[1][ks]
                            if ranks[0][ks] > 10 :         
                                step_hit10 += 1
                                step_hit3 += 1
                                step_hit1 += 1
                            else:
                                is_hit10 = 1.0 if ranking <= 10 else 0.0
                                if ranks[0][ks] > 3 :           # 
                                    step_hit3 += 1
                                    step_hit1 += 1
                                else:
                                    is_hit3 = 1.0 if ranking <= 3.0 else 0.0
                                    if ranks[0][ks] >1:
                                        step_hit1 += 1
                                    else:
                                        is_hit1 = 1.0 if ranking <=1 else 0.0
                        elif relation == 'asymmetry':
                            ranking = ranks[1][ks] 
                            if ranks[0][ks] > 10 :   
                                step_hit10 += 1
                                step_hit3 += 1
                                step_hit1 += 1
                            else:
                                is_hit10 = 1.0 if ranking > hit_false else 0.0 
                                if ranks[0][ks] > 3 :            
                                    step_hit3 += 1
                                    step_hit1 += 1
                                else:
                                    is_hit3 = 1.0 if ranking > hit_false else 0.0
                                    if ranks[0][ks] >1:
                                        step_hit1 += 1
                                    else:
                                        is_hit1 = 1.0 if ranking > hit_false else 0.0                      
                        elif relation == 'transitive' or relation == 'composition':
                            ranking = ranks[2][ks]
                            if ranks[0][ks] > 10 or ranks[1][ks] > 10:     
                                step_hit10 += 1
                                step_hit3 += 1
                                step_hit1 += 1
                            else:
                                is_hit10 = 1.0 if ranking <= 10 else 0.0
                                if ranks[0][ks] > 3 or ranks[1][ks] > 3:           # 
                                    step_hit3 += 1
                                    step_hit1 += 1
                                else:
                                    is_hit3 = 1.0 if ranking <= 3.0 else 0.0
                                    if ranks[0][ks] >1 or ranks[1][ks] > 1:
                                        step_hit1 += 1
                                    else:
                                        is_hit1 = 1.0 if ranking <=1 else 0.0
                        logs.append({
                                'HITS@1': is_hit1,
                                'HITS@3': is_hit3,
                                'HITS@10': is_hit10,
                        })                   

        metrics={}
        if len(logs) == 0:
            return metrics
        for metric in logs[0].keys():
            metrics[metric] = sum([log[metric] for log in logs])
        metrics['HITS@1'] =  metrics['HITS@1']/(test_num)
        metrics['HITS@3'] = metrics['HITS@3']/(test_num)
        metrics['HITS@10'] = metrics['HITS@10']/(test_num)
        metrics['step_hit1'] = step_hit1
        metrics['step_hit3'] = step_hit3
        metrics['step_hit10'] = step_hit10
        metrics['total_num'] = test_num
        return metrics


    #
    @staticmethod
    def class_test_step_print_case(model, test_triples, all_true_triples, relation, args, ndcg_caculate=False):
        '''
        Evaluate the model on test or valid datasets
        '''
        model.eval()
        size_triples = len(test_triples)
        # test_triples = [test_triples[0][0:10], test_triples[1][0:10]]
        test_dataloader_head= DataLoader(
                MulTestDataset(
                    test_triples, 
                    all_true_triples, 
                    args.nentity, 
                    args.nrelation, 
                    'h_rt'
                ), 
                batch_size=4,
                num_workers=1
            )
        test_dataloader_tail = DataLoader(
                MulTestDataset(
                    test_triples, 
                    all_true_triples, 
                    args.nentity, 
                    args.nrelation, 
                    'hr_t'
                ), 
                batch_size=4,
                num_workers=1
            )
        test_dataset_list = [test_dataloader_head, test_dataloader_tail]

        logs = []
        step = 0
        total_steps = sum([len(dataset) for dataset in test_dataset_list])
        test_num = 0
        step_hit10 = 0
        step_hit1 = 0
        step_hit3 = 0
        hit_false = 50
        test_case_list = []
        print("Total Step: ",total_steps)
        triple_embedding_list = []

        with torch.no_grad():
            for test_dataset in test_dataset_list:
                for positive_samples, negative_samples, filter_biass, mode in test_dataset:
                    step +=1
                    if step % 1000 == 0:
                        logging.info("ClassTest %d / %d" % (step, total_steps))
                    if args.cuda:
                        positive_samples = positive_samples.cuda()
                        negative_samples = negative_samples.cuda()
                        filter_biass = filter_biass.cuda()
                    positive_samples = list(positive_samples.split(1,1))
                    negative_samples = list(negative_samples.split(1,1))
                    filter_biass = list(filter_biass.split(1,1))
                    ranks=[[] for i in range(size_triples)]
                    mode = mode[0]
                    argsort_list = []
                    score_list = []
                    pos_list = []
                    for triple_sort in range(size_triples):
                        positive_sample = torch.squeeze(positive_samples[triple_sort],dim=1) 
                        negative_sample = torch.squeeze(negative_samples[triple_sort],dim=1)
                        filter_bias = torch.squeeze(filter_biass[triple_sort],dim=1)
                        batch_size = positive_sample.size(0)
                        h = positive_sample[:,0]
                        r = positive_sample[:,1]
                        t = positive_sample[:,2]
                        if mode == 'hr_t':
                            score = model(h,r, negative_sample, mode=mode)
                        else:
                            score = model(negative_sample,r,t, mode=mode)
                        score += filter_bias
                        score_list.append(score)
                        argsort = torch.argsort(score, dim = 1, descending=True)
                        argsort_list.append(argsort)
                        if mode == 'h_rt':
                            positive_arg = positive_sample[:, 0]
                        elif mode == 'hr_t':
                            positive_arg = positive_sample[:, 2]
                        else:
                            raise ValueError('mode %s not supported' % mode)
                        pos_list.append(positive_arg)
                        for i in range(batch_size):
                            ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                            assert ranking.size(0) == 1
                            ranking = 1 + ranking.item()    
                            ranks[triple_sort].append(ranking)
                    
                    for ks in range(len(ranks[0])):
                        ranking = None
                        is_hit10 = 0.0
                        is_hit3 = 0.0
                        is_hit1 = 0.0
                        test_num += 1


                        if relation == 'symmetry' or relation == 'inverse':

                            # head_emb = model.entity_embedding(h[ks])
                            # rel_emb = model.relation_embedding(r[ks])
                            # tail_emb = model.entity_embedding(t[ks])
                            # embedding_dict = {
                            #     "head":head_emb.detach().cpu().numpy().tolist(),
                            #     "rel_emb":rel_emb.detach().cpu().numpy().tolist(),
                            #     "tail_emb":tail_emb.detach().cpu().numpy().tolist()
                            # }
                            # triple_embedding_list.append(embedding_dict)

                            test_case = {}
                            test_case['mode'] = mode
                            positive_sample = torch.squeeze(positive_samples[0],dim=1) 
                            negative_sample = torch.squeeze(negative_samples[0],dim=1)
                            argsort = argsort_list[0]
                            positive_arg = pos_list[0]
                            score = score_list[0]
                            test_case['pre_truth'] = (positive_sample[ks,0].item(),positive_sample[ks,1].item(),positive_sample[ks,2].item())
                            test_case['pre_top10'] = negative_sample[ks][argsort[ks,:10]].cpu().numpy().tolist()
                            test_case['pre_Topscore'] = score[ks][argsort[ks,:10]].cpu().numpy().tolist()
                            test_case['pre_trueScore'] = (score[ks][positive_arg[ks]].cpu().numpy().tolist())
                            test_case['pre_trueRank'] = ranks[0][ks]
                            test_case['pre_max'] = torch.max(score[ks]).item()
                            test_case['pre_min'] = torch.min(score[ks]).item()
                            test_case['pre_mean'] = torch.mean(score[ks]).item()

                            # if ranks[0][ks] != 1:
                            #     test_case['pre_gt_pre_score'] = score[ks][argsort[ks,ranks[0][ks]-2]].item()
                            # else:
                            #     test_case['pre_gt_pre_score'] = (score[ks][positive_arg[ks]].cpu().numpy().tolist())
                            
                            # test_case['pre_gt_ne_score'] = score[ks][argsort[ks,ranks[0][ks]]].item()


                            positive_sample = torch.squeeze(positive_samples[1],dim=1) 
                            negative_sample = torch.squeeze(negative_samples[1],dim=1)
                            argsort = argsort_list[1]
                            positive_arg = pos_list[1]
                            score = score_list[1]
                            test_case['ne_truth'] = (positive_sample[ks,0].item(),positive_sample[ks,1].item(),positive_sample[ks,2].item())
                            test_case['ne_top10'] = negative_sample[ks][argsort[ks,:10]].cpu().numpy().tolist()
                            test_case['ne_Topscore'] = score[ks][argsort[ks,:10]].cpu().numpy().tolist()
                            test_case['ne_trueScore'] = (score[ks][positive_arg[ks]].cpu().numpy().tolist())
                            test_case['ne_trueRank'] = ranks[1][ks]

                            test_case['ne_max'] = torch.max(score[ks]).item()
                            test_case['ne_min'] = torch.min(score[ks]).item()
                            test_case['ne_mean'] = torch.mean(score[ks]).item()

                            # if ranks[1][ks] != 1:
                            #     test_case['ne_gt_pre_score'] = score[ks][argsort[ks,ranks[1][ks]-2]].item()
                            # elif ranks[1][ks] < score[ks].shape[-1]:
                            #     test_case['ne_gt_pre_score'] = (score[ks][positive_arg[ks]].cpu().numpy().tolist())
                            # test_case['ne_gt_ne_score'] = score[ks][argsort[ks,ranks[1][ks]]-1].item()

                            test_case_list.append(test_case)
                            ranking = ranks[1][ks]
                            if ranks[0][ks] > 10 :         
                                step_hit10 += 1
                                step_hit3 += 1
                                step_hit1 += 1
                            else:
                                is_hit10 = 1.0 if ranking <= 10 else 0.0
                                if ranks[0][ks] > 3 :           # 
                                    step_hit3 += 1
                                    step_hit1 += 1
                                else:
                                    is_hit3 = 1.0 if ranking <= 3.0 else 0.0
                                    if ranks[0][ks] >1:
                                        step_hit1 += 1
                                    else:
                                        is_hit1 = 1.0 if ranking <=1 else 0.0
                        elif relation == 'asymmetry':
                            test_case = {}
                            test_case['mode'] = mode
                            positive_sample = torch.squeeze(positive_samples[0],dim=1) 
                            negative_sample = torch.squeeze(negative_samples[0],dim=1)
                            argsort = argsort_list[0]
                            positive_arg = pos_list[0]
                            score = score_list[0]
                            test_case['pre_trueScore'] = (score[ks][positive_arg[ks]].cpu().numpy().tolist())
                            
                            positive_sample = torch.squeeze(positive_samples[1],dim=1) 
                            negative_sample = torch.squeeze(negative_samples[1],dim=1)
                            argsort = argsort_list[1]
                            positive_arg = pos_list[1]
                            score = score_list[1]
                            test_case['ne_trueScore'] = (score[ks][positive_arg[ks]].cpu().numpy().tolist())

                            test_case_list.append(test_case)
                            ranking = ranks[1][ks] 
                            if ranks[0][ks] > 10 :   
                                step_hit10 += 1
                                step_hit3 += 1
                                step_hit1 += 1
                            else:
                                is_hit10 = 1.0 if ranking > hit_false else 0.0 
                                if ranks[0][ks] > 3 :            
                                    step_hit3 += 1
                                    step_hit1 += 1
                                else:
                                    is_hit3 = 1.0 if ranking > hit_false else 0.0
                                    if ranks[0][ks] >1:
                                        step_hit1 += 1
                                    else:
                                        is_hit1 = 1.0 if ranking > hit_false else 0.0                      
                        elif relation == 'transitive' or relation == 'composition':
                            ranking = ranks[2][ks]
                            if ranks[0][ks] > 10 or ranks[1][ks] > 10:     
                                step_hit10 += 1
                                step_hit3 += 1
                                step_hit1 += 1
                            else:
                                is_hit10 = 1.0 if ranking <= 10 else 0.0
                                if ranks[0][ks] > 3 or ranks[1][ks] > 3:           # 
                                    step_hit3 += 1
                                    step_hit1 += 1
                                else:
                                    is_hit3 = 1.0 if ranking <= 3.0 else 0.0
                                    if ranks[0][ks] >1 or ranks[1][ks] > 1:
                                        step_hit1 += 1
                                    else:
                                        is_hit1 = 1.0 if ranking <=1 else 0.0
                        logs.append({
                                'HITS@1': is_hit1,
                                'HITS@3': is_hit3,
                                'HITS@10': is_hit10,
                        })                   

        metrics={}
        if len(logs) == 0:
            return metrics
        for metric in logs[0].keys():
            metrics[metric] = sum([log[metric] for log in logs])
        metrics['HITS@1'] =  metrics['HITS@1']/(test_num)
        metrics['HITS@3'] = metrics['HITS@3']/(test_num)
        metrics['HITS@10'] = metrics['HITS@10']/(test_num)
        metrics['step_hit1'] = step_hit1
        metrics['step_hit3'] = step_hit3
        metrics['step_hit10'] = step_hit10
        metrics['total_num'] = test_num
        return metrics,test_case_list,triple_embedding_list
 
    # 可以输出测试的loss，但是目前仅支持Pair Loss.
    @staticmethod
    def test_step(model, test_triples, all_true_triples, args, loss_function=None,ndcg_caculate=False):
        '''
        Evaluate the model on test or valid datasets
        '''
        model.eval()
        test_dataloader_head = DataLoader(
            TestDataset(
                test_triples, 
                all_true_triples, 
                args.nentity, 
                args.nrelation, 
                'h_rt'
            ), 
            batch_size=args.test_batch_size,
            num_workers=1, 
            collate_fn=TestDataset.collate_fn
        )
        test_dataloader_tail = DataLoader(
            TestDataset(
                test_triples, 
                all_true_triples, 
                args.nentity, 
                args.nrelation, 
                'hr_t'
            ), 
            batch_size=args.test_batch_size,
            num_workers=1, 
            collate_fn=TestDataset.collate_fn
        )
        test_dataset_list = [test_dataloader_head, test_dataloader_tail]
        logs = []
        step = 0
        total_steps = sum([len(dataset) for dataset in test_dataset_list])
        pos_loss_list = []
        neg_loss_list = []
        all_loss_list = []
        ndcg_at_1 = 0
        with torch.no_grad():
            for test_dataset in test_dataset_list:
                for positive_sample, negative_sample, filter_bias, mode in test_dataset:
                    batch_size = positive_sample.size(0)
                    if args.cuda:
                        positive_sample = positive_sample.cuda()
                        negative_sample = negative_sample.cuda()
                        filter_bias = filter_bias.cuda()
                        
                    h = positive_sample[:,0]
                    r = positive_sample[:,1]
                    t = positive_sample[:,2] 
                    if mode == 'hr_t':
                        negative_score = model(h,r, negative_sample, mode=mode)
                    else:
                        negative_score = model(negative_sample,r,t, mode=mode)
                    
                    # if loss_function != None:
                    #     positive_score = model(h,r, t)
                    #     score = negative_score + filter_bias*10
                    #     pos_loss,neg_loss = loss_function(positive_score,score,return_split=True)
                    #     pos_loss_list.append(pos_loss.item())
                    #     neg_loss_list.append(neg_loss.item())
                    #     all_loss_list.append((pos_loss.item() + neg_loss.item())/2)

                    score = negative_score + filter_bias
                    argsort = torch.argsort(score, dim = 1, descending=True)
                    if mode == 'h_rt':
                        positive_arg = h
                    elif mode == 'hr_t':
                        positive_arg = t
                    else:
                        raise ValueError('mode %s not supported' % mode)
                    
                    for i in range(batch_size):
                        ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                        assert ranking.size(0) == 1
                        ranking = 1 + ranking.item()
                        if ndcg_caculate:
                            ndcg_at_1 += ndcg_at_k(ranking)

                        logs.append({
                            'MRR': 1.0/ranking,
                            'MR': float(ranking),
                            'HITS@1': 1.0 if ranking <= 1 else 0.0,
                            'HITS@3': 1.0 if ranking <= 3 else 0.0,
                            'HITS@10': 1.0 if ranking <= 10 else 0.0,
                        })
                    if step % args.test_log_steps == 0:
                        logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))
                    step += 1
        

        logging.info("Avg Pos Loss %f", np.mean(pos_loss_list))
        logging.info("Avg Neg Loss %f", np.mean(neg_loss_list)) 
        logging.info("Avg all Loss %f", np.mean(all_loss_list))            
        
        metrics = {}
        for metric in logs[0].keys():
            metrics[metric] = sum([log[metric] for log in logs])/len(logs)
        if ndcg_caculate:
            return metrics, ndcg_at_1/len(logs)
        else:
            return metrics

