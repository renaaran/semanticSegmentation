#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 17:59:25 2020

@author: Renato B. Arantes
"""

import unittest
from resnet_seg import FacadeDataset
from torch.utils.data import DataLoader

class TestRestNetSeg(unittest.TestCase):

    ROOT_PATH = '../datasets/facades'

    def test_dataset_len(self):
        dataset = FacadeDataset(self.ROOT_PATH, root='train')
        self.assertEqual(len(dataset), 302)
        dataset = FacadeDataset(self.ROOT_PATH, root='test')
        self.assertEqual(len(dataset), 76)

    def test_train_dataset_read(self):
        dataset = FacadeDataset(self.ROOT_PATH, root='train')
        loader = DataLoader(dataset, shuffle=True)
        for x, y in loader:
            self.assertEqual(x.shape[-2:], y.shape[-2:])

    def test_test_dataset_read(self):
        dataset = FacadeDataset(self.ROOT_PATH, root='test')
        loader = DataLoader(dataset, shuffle=True)
        for x, y in loader:
            self.assertEqual(x.shape[-2:], y.shape[-2:])
            self.assertGreaterEqual(y.min(), 0)
            self.assertLessEqual(y.max(), 11)

if __name__ == '__main__':
    unittest.main()