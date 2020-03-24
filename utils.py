#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 11:53:43 2020

@author: Renato B. Arantes
"""
import torch
import numpy as np
import random

import torch.backends.cudnn as cudnn
from sklearn.metrics import jaccard_score

def initialize_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def initialize_torch(cuda=0):
    ngpu = torch.cuda.device_count()
    print('ngpus=%d' %(ngpu))
    print('torch.cuda.is_available=%d' % (torch.cuda.is_available()))
    if torch.cuda.is_available():
        print('torch.version.cuda=%s' % (torch.version.cuda))
    # Decide which device we want to run on
    device = torch.device(f'cuda:{cuda}' if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    print(device, device.type)
    ######################
    cudnn.benchmark = True
    return device, ngpu

def calc_iou(y_pred, y_true):
     return jaccard_score(y_pred.cpu().numpy().flatten(),
                          y_true.cpu().numpy().flatten(), average='micro')

