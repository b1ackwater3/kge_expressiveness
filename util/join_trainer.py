## 从本体层和联合训练

# 首先进行本体层训练

# 然后进行实体层训练

# 采用RotatE + r 来表示

# 这里需要实现一个整个模型的部分

# 构造四个数据集：增加reverse 然后单项实现的

# 设计两种训练策略
# 4个轮流训练

from util.data_process import DataProcesser as DP
import os
from core.TransER import TransER
import torch
from torch.optim.lr_scheduler import StepLR
from util.dataloader import NagativeSampleDataset,MultiShotItertor
from loss import NSSAL
from util.tools import logset
from util.model_util import ModelUtil
from torch.utils.data import DataLoader



def train_step(train_iterator,model,level,loss_function,cuda,optimizer):
    positive_sample,negative_sample, subsampling_weight, mode = next(train_iterator)
    if cuda:
        positive_sample = positive_sample.cuda()
        negative_sample = negative_sample.cuda()
        subsampling_weight = subsampling_weight.cuda()
    h = positive_sample[:,0]
    r = positive_sample[:,1]
    t = positive_sample[:,2]
    negative_score = model(h,r, negative_sample, mode=mode,level=level)
    positive_score = model(h,r,t,level=level)
    loss = loss_function(positive_score, negative_score,subsampling_weight)
    loss.backward()
    optimizer.step()
    log = {
        level+'_loss': loss.item()
    }
    return log

def logging_log(step, logs):
    metrics = {}
    for metric in logs[0].keys():
        metrics[metric] = sum([log[metric] for log in logs])/len(logs)
    logset.log_metrics('Training average', step, metrics)

def __main__():
    # 读取4个数据集
    cuda = True
    init_step = 0
    max_step = 100000
    save_steps = 10000
    root_path = "/home/skl/yl/models/JOIE_R"
    batch_size = 512
    logset.set_logger(root_path)

    if not os.path.exists(root_path):
        os.makedirs(root_path)

    data_path = ""
    ins = DP(os.path.join(data_path,"ins"),idDict=False, reverse=True)
    on = DP(os.path.join(data_path,"on"),idDict=False, reverse=True)
    cross = DP(os.path.join(data_path,"cross"),idDict=False, reverse=True)
    level = DP(os.path.join(data_path,"level"),idDict=False, reverse=True)

    n_size = 128
    ins_train = DataLoader(NagativeSampleDataset(ins.train,ins.nentity,ins.nrelation,n_size,'hr_t'),
        batch_size=batch_size,
        shuffle=True, 
        num_workers=max(1, 4//2),
        collate_fn=NagativeSampleDataset.collate_fn
    )
    on_train = DataLoader(NagativeSampleDataset(on.train,on.nentity,on.nrelation,n_size,'hr_t'),
            batch_size=batch_size,
            shuffle=True, 
            num_workers=max(1, 4//2),
            collate_fn=NagativeSampleDataset.collate_fn
        )
    cross_train = DataLoader(
            NagativeSampleDataset(cross.train,cross.nentity,cross.nrelation,n_size,'hr_t'),
            batch_size=batch_size,
            shuffle=True, 
            num_workers=max(1, 4//2),
            collate_fn=NagativeSampleDataset.collate_fn)

    level_train = DataLoader(
            NagativeSampleDataset(level.train,level.nentity,level.nrelation,n_size,'hr_t'),
            batch_size=batch_size,
            shuffle=True, 
            num_workers=max(1, 4//2),
            collate_fn=NagativeSampleDataset.collate_fn)
    train_iterator = MultiShotItertor(ins_train,on_train, cross_train, level_train)

    # 构造模
    on_dim = 200
    in_dim = 200
    lr = 0.0001
    model = TransER(on.nentity,on.nrelation,on_dim,ins.nentity,ins.nrelation,in_dim)
    # 设置优化器
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=lr
    )
    # 设置学习率更新策略

    # 构造训练器

    # 训练器：每个step里面轮流房
    loss_function_in = NSSAL(4)
    loss_function_on = NSSAL(6)
    loss_function_type = NSSAL(10,False)
    loss_function_level = NSSAL(10,False)
    log_steps = 100
    logsA = []
    logsB = []
    logsC = []
    logsD = []
    for step in range(init_step, max_step):
        log1 = train_step(train_iterator,model,"in",loss_function_in,cuda,optimizer)
        log2 = train_step(train_iterator,model,"on",loss_function_on,cuda,optimizer)
        log3 = train_step(train_iterator,model,"cross",loss_function_type,cuda,optimizer)
        log4 = train_step(train_iterator,model,"level",loss_function_level,cuda,optimizer)
        logsA.append(log1)
        logsB.append(log2)
        logsC.append(log3)
        logsD.append(log4)
        if step % log_steps == 0:
            logging_log(step,logsA)
            logging_log(step,logsB)
            logging_log(step,logsC)
            logging_log(step,logsD)
            logsA = []
            logsB = []
            logsC = []
            logsD = []
        if step % save_steps == 0:
            ModelUtil.save_model(model,optimizer,save_variable_list={"lr":lr},path=root_path)
        

