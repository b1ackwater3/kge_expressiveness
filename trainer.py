import numpy as np
import torch
from util.dataloader import OneToNDataset,SimpleTripleDataset
from util.dataloader import TestDataset
from util.dataloader import OneShotIterator
from torch.utils.data import DataLoader
import logging
import torch.nn.functional as F
from regularizers import N3
from sklearn.metrics import average_precision_score

from util.tools import logset

class NagativeSampleTrainer():
    def __init__(self,args):
        super().__init__()

    @staticmethod
    def train_step(model, optimizer, train_iterator, args):
        '''
        A single train step. Apply back-propation and return the loss
        '''
        Loss_FUNCTION = F.logsigmoid

        model.train()
        optimizer.zero_grad()
        positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)

        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()

        negative_score, _ = model((positive_sample, negative_sample), mode=mode)

        if args.negative_adversarial_sampling:
            negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim = 1).detach() 
                              * Loss_FUNCTION(-negative_score)).sum(dim = 1)
        else:
            negative_score = Loss_FUNCTION(-negative_score).mean(dim = 1)


        positive_score,_ = model(positive_sample)
        positive_score = Loss_FUNCTION(positive_score).squeeze(dim = 1)
        
        if args.uni_weight:
            positive_sample_loss = - positive_score.mean()
            negative_sample_loss = - negative_score.mean()
        else:
            positive_sample_loss = - (subsampling_weight * positive_score).sum()/subsampling_weight.sum()
            negative_sample_loss = - (subsampling_weight * negative_score).sum()/subsampling_weight.sum()


        loss = (positive_sample_loss + negative_sample_loss)/2
        if model.model_name == 'RotPro' and args.constrains:
            a1 = model.projection_embedding_a - 1.0
            a0 = model.projection_embedding_a - 0.0
            a = torch.abs(a1 * a0)
            penalty = torch.ones_like(a)
            penalty[a > args.gamma_m] = args.beta
            l_a = (a * penalty).norm(p=2)

            b1 = model.projection_embedding_b - 1.0
            b0 = model.projection_embedding_b - 0.0
            b = torch.abs(b1 * b0)
            penalty = torch.ones_like(b)
            penalty[b > args.gamma_m] = args.beta
            l_b = (b * penalty).norm(p=2)
            loss += (l_a + l_b) * args.alpha

        if args.regularization != 0.0:
            #Use L3 regularization for ComplEx and DistMult
            # regularization = args.regularization * (
            #     model.entity_embedding.norm(p = 3)**3 + 
            #     model.relation_embedding.norm(p = 3).norm(p = 3)**3
            # )
            #Use N3 regularization for ComplEx
            re_entity, im_entity = torch.chunk(model.entity_embedding, 2, dim=1)
            re_relation, im_relation = torch.chunk(model.relation_embedding, 2, dim=1)
            factor_relation =torch.sqrt(re_relation**2 + im_relation**2)
            factor_entity = torch.sqrt(re_entity**2 + im_entity**2)
            # 实现 N3 正则 for ComplEx
            regularization = args.regularization * (
                torch.sum(factor_relation**3)/model.nrelation + torch.sum(factor_entity**3)/model.nentity
            ) 

            loss = loss + regularization
            regularization_log = {'regularization': regularization.item()}
        else:
            regularization_log = {}
            
        loss.backward()

        optimizer.step()

        log = {
            **regularization_log,
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item()
        }

        return log

    @staticmethod
    def test_step(model, test_triples, all_true_triples, args):
        '''
        Evaluate the model on test or valid datasets
        '''
        model.eval()
        if args.countries:
            #Countries S* datasets are evaluated on AUC-PR
            #Process test data for AUC-PR evaluation
            sample = list()
            y_true  = list()
            for head, relation, tail in test_triples:
                for candidate_region in args.regions:
                    y_true.append(1 if candidate_region == tail else 0)
                    sample.append((head, relation, candidate_region))

            sample = torch.LongTensor(sample)
            if args.cuda:
                sample = sample.cuda()

            with torch.no_grad():
                result,_  = model(sample)
                y_score = result.squeeze(1).cpu().numpy()

            y_true = np.array(y_true)

            #average_precision_score is the same as auc_pr
            auc_pr = average_precision_score(y_true, y_score)

            metrics = {'auc_pr': auc_pr}
            
        else:
            test_dataloader_head = DataLoader(
                TestDataset(
                    test_triples, 
                    all_true_triples, 
                    args.nentity, 
                    args.nrelation, 
                    'head-batch'
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
                    'tail-batch'
                ), 
                batch_size=args.test_batch_size,
                num_workers=1, 
                collate_fn=TestDataset.collate_fn
            )
            
            test_dataset_list = [test_dataloader_head, test_dataloader_tail]
            logs = []

            step = 0
            total_steps = sum([len(dataset) for dataset in test_dataset_list])

            with torch.no_grad():
                for test_dataset in test_dataset_list:
                    # 下面这个循环构建多重
                    for positive_sample, negative_sample, filter_bias, mode in test_dataset:
                        # 做了负采样
                        batch_size = positive_sample.size(0)
                        if args.cuda:
                            positive_sample = positive_sample.cuda()
                            negative_sample = negative_sample.cuda()
                            filter_bias = filter_bias.cuda()

                        score,_ = model((positive_sample, negative_sample), mode)
                        score += filter_bias

                        #Explicitly sort all the entities to ensure that there is no test exposure bias
                        argsort = torch.argsort(score, dim = 1, descending=True)

                        if mode == 'head-batch':
                            positive_arg = positive_sample[:, 0]
                        elif mode == 'tail-batch':
                            positive_arg = positive_sample[:, 2]
                        else:
                            raise ValueError('mode %s not supported' % mode)
                        
                        for i in range(batch_size):
                            #Notice that argsort is not ranking
                            ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                            assert ranking.size(0) == 1
                            #ranking + 1 is the true ranking used in evaluation metrics
                            ranking = 1 + ranking.item()
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
            metrics = {}
            for metric in logs[0].keys():
                metrics[metric] = sum([log[metric] for log in logs])/len(logs)
        return metrics



class OneToNTrainer():
    def __init__(self,model, optimizer, train_triples, nentity, nrelation, args):
        super().__init__()
        
        self.model = model
        self.opt = optimizer
        self.label_smoothing = 0.0
        self.train_mode = args.train_mode
        if self.train_mode == '1to1':
            dataset =  SimpleTripleDataset(train_triples,nentity,nrelation)
        else:
            dataset =  OneToNDataset(train_triples,nentity,nrelation)
            self.label_smoothing = args.label_smoothing

        loader = DataLoader(dataset,
            batch_size=args.batch_size,
            shuffle=True, 
            num_workers=max(1, args.cpu_num//2)
        )
        self.dataset  =OneShotIterator(loader)
        
        if model.model_name == 'TuckER':
            self.loss = torch.nn.BCELoss()
        elif model.model_name == 'ComplEx':
            self.loss = torch.nn.CrossEntropyLoss(reduction='mean')
            
        self.args = args
        if args.regularization != 0.0:
            self.regularizer = N3(args.regularization)

    def train_step(self):
        '''
        A single train step. Apply back-propation and return the loss
        '''
        self.model.train()
        self.opt.zero_grad()
        if self.train_mode == '1toN':
            data, ground_true = next(self.dataset)
        else:
            data = next(self.dataset)
            ground_true = data[:,2]
        
        if self.label_smoothing != 0.0:
            ground_true = ((1.0-self.label_smoothing)*ground_true) + (1.0/ground_true.size(1))  
        
        if self.args.cuda:
            data = data.cuda()
            ground_true = ground_true.cuda()

        score, factor = self.model(data, mode=self.train_mode)
        loss = self.loss(score, ground_true)

        if self.args.regularization != 0.0:
            reg = self.regularizer.forward(factor)
            loss += reg

        loss.backward()
        self.opt.step()
        log = {
            'loss': loss.item()
        }
        return log

    def test_step(self,test_triples, all_true_triples, args):
        self.model.eval()
        # # 构造test 数据集
        # test_dataloader_head = DataLoader(
        #     TestDataset(
        #         test_triples, 
        #         all_true_triples, 
        #         args.nentity, 
        #         args.nrelation, 
        #         'head-batch'
        #     ), 
        #     batch_size=args.test_batch_size,
        #     num_workers=1, 
        #     collate_fn=TestDataset.collate_fn
        # )
        test_dataloader_tail = DataLoader(
            TestDataset(
                test_triples, 
                all_true_triples, 
                args.nentity, 
                args.nrelation, 
                'tail-batch'
            ), 
            batch_size=args.test_batch_size,
            num_workers=1, 
            collate_fn=TestDataset.collate_fn
        )
        test_dataset_list = [test_dataloader_tail]
        logs = []
        step = 0
        total_steps = sum([len(dataset) for dataset in test_dataset_list])
        with torch.no_grad():
            for test_dataset in test_dataset_list:
                # 下面这个循环构建多重
                for positive_sample, negative_sample, filter_bias, mode in test_dataset:
                    # 做了负采样
                    batch_size = positive_sample.size(0)
                    if args.cuda:
                        positive_sample = positive_sample.cuda()
                        negative_sample = negative_sample.cuda()
                        filter_bias = filter_bias.cuda()
                    score,_ = self.model((positive_sample, negative_sample), mode)
                    score += filter_bias
                    #Explicitly sort all the entities to ensure that there is no test exposure bias
                    argsort = torch.argsort(score, dim = 1, descending=True)
                    if mode == 'head-batch':
                        positive_arg = positive_sample[:, 0]
                    elif mode == 'tail-batch':
                        positive_arg = positive_sample[:, 2]
                    else:
                        raise ValueError('mode %s not supported' % mode)
                    
                    # 这里的rank 直接记录下来
                    for i in range(batch_size):
                        #Notice that argsort is not ranking
                        ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                        assert ranking.size(0) == 1
                        #ranking + 1 is the true ranking used in evaluation metrics
                        ranking = 1 + ranking.item()
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
        metrics = {}
        for metric in logs[0].keys():
            metrics[metric] = sum([log[metric] for log in logs])/len(logs)
        return metrics


