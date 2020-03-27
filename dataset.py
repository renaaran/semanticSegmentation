#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 13:38:29 2020

@author: Renato B. Arantes
"""
import os
import torch
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

from albumentations import (
    HorizontalFlip,
    VerticalFlip,
    Transpose,
    RandomRotate90
)

from PIL import Image

NUM_LABELS = 12

class FacadeDataset(Dataset):

    LABELS = set([i for i in range(NUM_LABELS)])
    AUGMENTS = [RandomRotate90(p=1),
                HorizontalFlip(p=1),
                VerticalFlip(p=1),
                Transpose(p=1)]

    def __init__(self, root_dir, augment=False, root='train'):
        self.root_dir = root_dir
        self.root = root
        self.files_count = 0
        self.augment = augment
        for _ in os.listdir(os.path.join(self.root_dir, root)):
            self.files_count += 1
        assert self.files_count%2 == 0
        self.files_count //= 2
        self.preprocess = transforms.Compose([
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225]),
         ])

    def __len__(self):
        return self.files_count

    def __read(self, index, ext):
        file_name = f'{index}.{ext}'
        img_path = os.path.join(self.root_dir, self.root, file_name)
        return Image.open(img_path)

    def __transform(self, img, lbl):
        r = np.random.randint(0, len(self.AUGMENTS)+1)
        if r == len(self.AUGMENTS):
            return img, lbl
        else:
            aug = self.AUGMENTS[r](image=img, mask=lbl)
            return aug['image'], aug['mask']

    def __getitem__(self, index):
        img = np.array(self.__read(index+1, 'jpg'))
        lbl = np.array(self.__read(index+1, 'png'))-1
        if self.augment:
            img, lbl = self.__transform(img, lbl)
        img = self.preprocess(img)
        lbl = torch.tensor(lbl)
        assert set(np.unique(lbl)).issubset(self.LABELS), \
            print(index, set(np.unique(lbl)))
        return img, lbl.long()

