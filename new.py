#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 12:39:07 2020

Plot confusion matrix

@author: Renato B. Arantes
"""

import os
import torch
import torch.nn as nn
import itertools
import numpy as np
import argparse
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

from torch.utils.data import DataLoader
from dataset import FacadeDataset, GeoDataset
from utils import initialize_seeds, initialize_torch, calc_iou
from torchvision.models.segmentation import deeplabv3_resnet50, \
    deeplabv3_resnet101

plt.style.use('seaborn-whitegrid')

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', required=True, 
                    help='expriments results path')
opt = parser.parse_args()

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    with plt.style.context('default'):
        plt.figure(figsize=(10,10))
        plt.imshow(cm, cmap=cmap)
        plt.title(title)
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)
    
        fmt = '.4f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()
        plt.savefig('./cm.jpg');
    
def load_model(model_type, model_path, n):
    if model_type == 50:
        model = deeplabv3_resnet50(num_classes=GeoDataset.NUM_LABELS, 
                                   pretrained=False)
    elif model_type == 101:
        model = deeplabv3_resnet101(num_classes=GeoDataset.NUM_LABELS, 
                                    pretrained=False)
    else: raise Exception(f'Invalid model: {model_type}')
    model = model.to(device)
    if (device.type == 'cuda') and (ngpu >= 1):
        model = nn.DataParallel(model, list(range(ngpu)))
    checkpoint = torch.load(os.path.join(model_path, f'model_{n}.pth'))
    model.load_state_dict(checkpoint)
    model.eval()
    return model

def load_test_dataset():
    test_dataset = GeoDataset(
        os.path.join(opt.dataroot, 'geo+minor'), root='test')
    print(f'test_dataset size={len(test_dataset)}')
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=1)
    return test_loader

def evaluate_test_dataset(exp, dataset, classes, N=5):
    _y_pred = np.array([])
    _y = np.array([])
    print(f'******* {dataset}')
    miou = []
    path = os.path.join(opt.dataroot,
                dataset,
               'experiments',
               exp)
    model_type = 101 if 'ResNet101' in exp else 50
    for i in [0]:
        model = load_model(model_type, path, i)
        test_loader = load_test_dataset()
        iou = []
        i = 0
        for X, y in test_loader:
            X = X.to(device)
            y = y.to(device)
            output = model(X)['out']
            probs = torch.functional.F.softmax(output, 1)
            y_pred = torch.argmax(probs, dim=1)
            iou.append(calc_iou(y_pred, y))
            i += 1
            print(i)
            _y = np.concatenate((_y, y.cpu().numpy().flatten()))
            _y_pred = np.concatenate((_y_pred, y_pred.cpu().numpy().flatten()))
        miou.append(np.mean(iou))
        print(miou[-1])
    print(miou)
    print(np.mean(miou), np.std(miou))
    cm = confusion_matrix(_y, _y_pred)
    print(cm.shape)
    plot_confusion_matrix(cm, normalize=True, 
                          classes=classes,
                          title=f'Confusion Matrix ({dataset})')

def calc_and_plot_confusion_matrix(exp):
    global device, ngpu
    initialize_seeds()
    device, ngpu = initialize_torch()
#    device = 'cpu'
    evaluate_test_dataset(exp, 'geo+minor', GeoDataset.LABELS)
    # evaluate_test_dataset(device, exp, 'facades+fake_B')
    # evaluate_test_dataset(device, exp, 'facades+rec_B')
    # evaluate_test_dataset(device, exp, 'facades+fake_b+rec_B')
    # evaluate_test_dataset(device, exp, 'facade_aug')
    # evaluate_test_dataset(device, exp, 'facade+fake_B_aug')
    
if __name__ == "__main__":
     calc_and_plot_confusion_matrix('ResNet101,lr=0.0002,epochs=50,earlystop')

