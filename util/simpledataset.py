from cmd import IDENTCHARS
from concurrent.futures import thread

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os


class SimpleTriplesDataset(Dataset):
    def __init__(self, triples):
        self.triples = triples
        
    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):

        return torch.LongTensor(list(self.triples[idx]))

    @staticmethod
    def collate_fn(data):
        triples = torch.stack(data, dim=0)
        return triples