#!/usr/bin/env python
# coding: utf-8

# Modified version of https://github.com/kaidic/LDAM-DRW

import torch

import numpy as np
import torch.nn.functional as F

from torch import nn

class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        m_list = np.divide(1.0, np.sqrt(np.sqrt(cls_num_list)), out=np.zeros(np.sqrt(np.sqrt(cls_num_list)).shape), where=np.sqrt(np.sqrt(cls_num_list))!=0)
        m_list = m_list * (max_m / np.max(m_list))
        self.m_list = torch.from_numpy(m_list).float().to(self.device)
        
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        index_float = index.type(torch.FloatTensor).to(self.device)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
    
        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s*output, target, weight=self.weight)
