

import torch
import torch.nn as nn
import math
import torch.nn.functional as F

# Margin ranking loss
class MRL(nn.Module):
    def __init__(self,gamma):
        super(MRL,self).__init__()
        self.gamma = torch.tensor(gamma,dtype=torch.float,requires_grad=False)

    def forward(self,positive_samples_score, negative_samples_score): 
        positive_samples_score = positive_samples_score.unsqueeze(dim = -1)
        return torch.relu(self.gamma - positive_samples_score + negative_samples_score).mean()

class MRL_plus(nn.Module):
    def __init__(self,gamma):
        super(MRL_plus,self).__init__()
        self.gamma = torch.tensor(gamma,dtype=torch.float,requires_grad=False)

    def forward(self,positive_samples_score, negative_samples_score,subsampling_weight=None): 
        weight =  (F.softmax(negative_samples_score,dim=-1)).detach()
        value = self.gamma - positive_samples_score + negative_samples_score
        weight_value =torch.sum(value*weight,dim=-1)
        if subsampling_weight != None:
            loss = (weight_value*subsampling_weight)/subsampling_weight.sum()
            return torch.relu(loss).mean()
        else:
            loss =torch.relu(weight_value).mean()
            return torch.relu(loss).mean()
            

# Self-adversarial negativesamplingloss
class NSSAL(nn.Module):
    def __init__(self, gamma=4.0, adversarial=True, adversarial_temperature=1.0,s_a_g=True):
        super(NSSAL, self).__init__()
        self.gamma = torch.tensor(gamma,dtype=torch.float,requires_grad=False,device='cuda')
        self.adversarial_temperature = torch.tensor(adversarial_temperature,dtype=torch.float,requires_grad=False)
        self.adversarial = adversarial
        self.s_a_g = s_a_g

    def forward(self, positive_samples_score, negative_samples_score,subsampling_weight=None,dis_weight=None,return_split=False):
        
        if dis_weight != None:
            negative_samples_score = negative_samples_score + dis_weight
        if self.s_a_g:
            positive_score = self.gamma + positive_samples_score
            negative_score = self.gamma + negative_samples_score
        else:
            positive_score = positive_samples_score
            negative_score = negative_samples_score
       
        positive_score = F.logsigmoid(positive_score).squeeze(dim = -1)
       
        if self.adversarial:
            weight = (F.softmax(negative_score*self.adversarial_temperature,dim=-1)).detach()
            negative_score = F.logsigmoid(-negative_score)
            negative_score = (weight * negative_score).sum(dim=1)
        else:
            negative_score = F.logsigmoid(-negative_score).mean(dim = 1)

        if subsampling_weight != None:
            positive_sample_loss = - (subsampling_weight * positive_score).sum()/subsampling_weight.sum()
            negative_sample_loss = - (subsampling_weight * negative_score).sum()/subsampling_weight.sum()
        else:
            positive_sample_loss = - positive_score.mean()
            negative_sample_loss = - negative_score.mean()
        if return_split:
            return positive_sample_loss,negative_sample_loss
        return (positive_sample_loss + negative_sample_loss)/2

class NSSAL_aug(nn.Module):
    def __init__(self, gamma=4.0, adversarial=True, adversarial_temperature=0.5, aug_weight=2):
        super(NSSAL_aug, self).__init__()
        self.gamma = torch.tensor(gamma,dtype=torch.float,requires_grad=False,device='cuda')
        self.adversarial_temperature = torch.tensor(adversarial_temperature,dtype=torch.float,requires_grad=False)
        self.adversarial = adversarial
        self.aug_weight = aug_weight

    def forward(self, positive_samples_score, negative_samples_score,subsampling_weight=None,dis_weight=None):
        

        positive_score = self.gamma + positive_samples_score
        negative_score = self.gamma + negative_samples_score
       
        positive_score = F.logsigmoid(positive_score).squeeze(dim = -1)
       
        if self.adversarial:
            weight = (F.softmax(negative_score*self.adversarial_temperature,dim=-1)).detach()
            negative_score = F.logsigmoid(-negative_score)
            negative_score = (weight * negative_score).sum(dim=1)
        else:
            negative_score = F.logsigmoid(-negative_score).mean(dim = 1)

        if subsampling_weight != None:
            positive_sample_loss = - (subsampling_weight * positive_score).sum()/subsampling_weight.sum()
            negative_sample_loss = - (subsampling_weight * negative_score).sum()/subsampling_weight.sum()
        else:
            positive_sample_loss = - positive_score.mean()
            negative_sample_loss = - negative_score.mean()
        return (positive_sample_loss + self.aug_weight*negative_sample_loss)/2


class NSSAL_sub(nn.Module):
    def __init__(self, gamma=4.0, adversarial=True, adversarial_temperature=0.5, aug_weight=2):
        super(NSSAL_sub, self).__init__()
        self.gamma = torch.tensor(gamma,dtype=torch.float,requires_grad=False,device='cuda')
        self.adversarial_temperature = torch.tensor(adversarial_temperature,dtype=torch.float,requires_grad=False)
        self.adversarial = adversarial
        self.aug_weight = aug_weight

    def forward(self, positive_samples_score, negative_samples_score,subsampling_weight=None,dis_weight=None):
        
        positive_score = self.gamma + positive_samples_score
        negative_score = self.gamma + negative_samples_score + self.aug_weight
        positive_score = F.logsigmoid(positive_score).squeeze(dim = -1)
        if self.adversarial:
            weight = (F.softmax(negative_score*self.adversarial_temperature,dim=-1)).detach()
            negative_score = F.logsigmoid(-negative_score)
            negative_score = (weight * negative_score).sum(dim=1)
        else:
            negative_score = F.logsigmoid(-negative_score).mean(dim = 1)
        if subsampling_weight != None:
            positive_sample_loss = - (subsampling_weight * positive_score).sum()/subsampling_weight.sum()
            negative_sample_loss = - (subsampling_weight * negative_score).sum()/subsampling_weight.sum()
        else:
            positive_sample_loss = - positive_score.mean()
            negative_sample_loss = - negative_score.mean()
        return (positive_sample_loss + negative_sample_loss)/2

class PLL(nn.Module):
    def __init__(self,):
        super(PLL, self).__init__()
    
    def forward(self,positive_samples_score, negative_samples_score):
        return F.softplus((negative_samples_score,positive_samples_score),dim=-1)