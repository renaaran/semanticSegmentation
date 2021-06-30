#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 13:38:29 2020

Definitions for the facade and geospatial datasets.

@author: Renato B. Arantes
"""
import os
import glob
import torch
import numpy as np
import torchvision.transforms as transforms

from torch.utils.data import Dataset
from random import shuffle

from PIL import Image


class BaseDataset(Dataset):
    def __init__(self, root_dir, root='train', mask='*.jpg'):
        self.root_dir = root_dir
        self.root = root
        self.files_count = 0
        self.files = []
        for file_path in glob.glob(os.path.join(self.root_dir, root, mask)):
            file_name = os.path.splitext(os.path.basename(file_path))[0]
            self.files.append(file_name)
        shuffle(self.files)
        self.files_count = len(self.files)

    def __len__(self):
        return self.files_count

    def _read(self, index, ext):
        file_name = f'{self.files[index]}.{ext}'
        img_path = os.path.join(self.root_dir, self.root, file_name)
        return Image.open(img_path)


class AugmentedDataset(BaseDataset):
    def __init__(self, root_dir, root='train', augments=None):
        super().__init__(root_dir, root)
        self.augments = augments
        if augments is not None:
            self.augments.append(A.Flip(p=0.0))
            self.aug_order = list(range(len(self.augments)))*self.files_count
            shuffle(self.aug_order)
            self.files_count *= len(self.augments)

    def _transform(self, index, img, lbl):
        #print(index, self.aug_order[index])
        index = self.aug_order[index]
        aug = self.augments[index](image=img, mask=lbl)
        return aug['image'], aug['mask']

    def _read(self, index, ext):
        #print(index, index % len(self.files))
        index = index % len(self.files)
        return super()._read(index, ext)

class BoxDataset(AugmentedDataset):

    NUM_LABELS = 2
    LABELS = set([i for i in range(NUM_LABELS)])

    def __init__(self, root_dir, augment=False, root='train'):
        if augment:
            super().__init__(root_dir, root=root,
                             augments=[A.RandomRotate90(p=1),
                                       A.HorizontalFlip(p=1),
                                       A.Transpose(p=1),
                                       A.RandomSizedCrop(min_max_height=(50, 50),
                                                       height=256,
                                                       width=256, p=1)])
        else:
            super().__init__(root_dir, root=root)
        self.preprocess = transforms.Compose([
             transforms.ToPILImage(),
             transforms.Resize((456, 456), interpolation=Image.NEAREST), # https://discuss.pytorch.org/t/transforms-resize-the-value-of-the-resized-pil-image/35372/2
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.4749, 0.4510, 0.4152],
                                   std=[0.2217, 0.2175, 0.2189]),
         ])

    def __getitem__(self, index):
        img = np.array(self._read(index, 'jpg'))
        lbl = np.array(self._read(index, 'png'))
        if self.augments is not None:
            img, lbl = self._transform(index, img, lbl)
        img = self.preprocess(img)
        lbl = torch.tensor(lbl)
        assert set(np.unique(lbl)).issubset(self.LABELS), \
            print(index, set(np.unique(lbl)))
        return img, lbl.long()

class SunDataset(AugmentedDataset):

    NUM_LABELS = 14
    LABELS = set([i for i in range(NUM_LABELS)])

    def __init__(self, root_dir, augment=False, root='train'):
        if augment:
            super().__init__(root_dir, root=root,
                             augments=[A.RandomRotate90(p=1),
                                       A.HorizontalFlip(p=1),
                                       A.Transpose(p=1),
                                       A.RandomSizedCrop(min_max_height=(50, 50),
                                                       height=256,
                                                       width=256, p=1)])
        else:
            super().__init__(root_dir, root=root)
        self.preprocess = transforms.Compose([
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.4749, 0.4510, 0.4152],
                                   std=[0.2217, 0.2175, 0.2189]),
         ])

    def __getitem__(self, index):
        img = np.array(self._read(index, 'jpg'))
        lbl = np.array(self._read(index, 'png'))
        if self.augments is not None:
            img, lbl = self._transform(index, img, lbl)
        img = self.preprocess(img)
        lbl = torch.tensor(lbl)
        assert set(np.unique(lbl)).issubset(self.LABELS), \
            print(index, set(np.unique(lbl)))
        return img, lbl.long()

class FacadesDataset(AugmentedDataset):

    NUM_LABELS = 12
    LABELS = set([i+1 for i in range(NUM_LABELS)])
    LABELS_NAME = ['Background',
                     'Facade',
                     'Window',
                     'Door',
                     'Cornice',
                     'Sill',
                     'Balcony',
                     'Blind',
                     'Deco',
                     'Molding',
                     'Pillar',
                     'Shop']

    def __init__(self, root_dir, augment=False, root='train'):
        if augment:
            super().__init__(root_dir, root=root,
                             augments=[A.RGBShift(p=1)])
        else:
            super().__init__(root_dir, root=root)
        self.preprocess = transforms.Compose([
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.4749, 0.4510, 0.4152],
                                   std=[0.2217, 0.2175, 0.2189]),
         ])

    def __getitem__(self, index):
        img = np.array(super()._read(index, 'jpg'))
        lbl = np.array(super()._read(index, 'png'))
        if self.augments is not None:
            img, lbl = self._transform(index, img, lbl)
        img = self.preprocess(img)
        lbl = torch.tensor(lbl)
        assert set(np.unique(lbl)).issubset(self.LABELS), \
            print(index, set(np.unique(lbl)))
        return img, lbl.long()

class GeoDataset(BaseDataset):

    NUM_LABELS = 6
    LABELS = set([i for i in range(NUM_LABELS)])

    NO_DAMAGE = "no-damage"
    MINOR_DAMAGE = "minor-damage"
    MAJOR_DAMAGE = "major-damage"
    DESTROYED = "destroyed"
    UN_CLASSIFIED = "un-classified"

    damage_types = {
        NO_DAMAGE: 1,
        MINOR_DAMAGE: 2,
        MAJOR_DAMAGE: 3,
        DESTROYED: 4,
        UN_CLASSIFIED: 5
    }

    def __init__(self, root_dir, augment=False, root='train'):
        super().__init__(root_dir, root)
        self.preprocess = transforms.Compose([
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.3269, 0.3545, 0.2604],
                                  std=[0.1292, 0.1135, 0.1101]),

         ])

    def __getitem__(self, index):
        img = np.array(super()._read(index, 'jpg'))
        lbl = np.array(super()._read(index, 'png'))
        img = self.preprocess(img)
        lbl = torch.tensor(lbl)
        assert set(np.unique(lbl)).issubset(self.LABELS), \
            print(index, set(np.unique(lbl)))
        return img, lbl.long()

    @staticmethod
    def get_un_classified_label():
        return GeoDataset.damage_types[GeoDataset.UN_CLASSIFIED]

class CityscapeDataset(AugmentedDataset):

    NUM_LABELS = 35
    LABELS = set([i for i in range(NUM_LABELS)])

    def __init__(self, root_dir, augment=False, root='train'):
        super().__init__(root_dir, root=root)
        self.preprocess = transforms.Compose([
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.28689554, 0.32513303, 0.28389177],
                                   std=[0.18696375, 0.19017339, 0.18720214]),
         ])

    def __getitem__(self, index):
        img = np.array(super()._read(index, 'jpg'))
        lbl = np.array(super()._read(index, 'png'))
        if self.augments is not None:
            img, lbl = self._transform(index, img, lbl)
        img = self.preprocess(img)
        lbl = torch.tensor(lbl)
        assert set(np.unique(lbl)).issubset(self.LABELS), \
            print(index, set(np.unique(lbl)))
        return img, lbl.long()
