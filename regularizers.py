from abc import ABC, abstractmethod
from typing import Tuple

import torch
from torch import nn


class N3(nn.Module):
    def __init__(self, weight: float):
        super(N3, self).__init__()
        self.weight = torch.tensor(weight,dtype=torch.float,requires_grad=False)

    def forward(self, factors):
        norm = 0
        for f in factors:
            norm += self.weight * torch.sum(
                torch.abs(f) ** 3
            )
        return norm / factors[0].shape[0]

class PLorm(nn.Module):
    def __init__(self,weight:float):
        super(PLorm, self).__init__()
        self.weight =torch.tensor(weight,dtype=torch.float,requires_grad=False)
        
    
    def forward(self, factors):
        result = torch.tensor(0)
        for factor in factors:
            result += factor.norm(3)**3
        return result*self.weight
