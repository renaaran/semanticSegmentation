#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 12:39:07 2020

@author: Renato B. Arantes
"""

import os
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from dataset import FacadeDataset, NUM_LABELS
from utils import initialize_seeds, initialize_torch, calc_iou
from torchvision.models.segmentation import deeplabv3_resnet50, \
    deeplabv3_resnet101

plt.style.use('seaborn-whitegrid')

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', required=True, help='expriments results path')
opt = parser.parse_args()

def load_model(device, model_type, model_path, n):
    if model_type == 50:
        model = deeplabv3_resnet50(num_classes=NUM_LABELS, pretrained=False)
    elif model_type == 101:
        model = deeplabv3_resnet101(num_classes=NUM_LABELS, pretrained=False)
    else: raise Exception(f'Invalid model: {model_type}')
    model = model.to(device)
    model.load_state_dict(torch.load(os.path.join(model_path, f'model_{n}.pth')))
    model.eval()
    return model

def load_test_dataset():
    test_dataset = FacadeDataset(
        os.path.join(opt.dataroot, 'extended'), root='test')
    print(f'test_dataset size={len(test_dataset)}')
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=1)
    return test_loader

def evaluate_test_dataset(device, exp, dataset, N=5):
    print(f'******* {dataset}')
    miou = []
    path = os.path.join(opt.dataroot,
                dataset,
               'experiments',
               exp)
    model_type = 101 if 'ResNet101' in exp else 50
    for i in range(N):
        model = load_model(device, model_type, path, i)
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
            if i == 76: break
        miou.append(np.mean(iou))
        print(miou[-1])
    print(miou)
    print(np.mean(miou), np.std(miou))

def evaluate_test(exp):
    initialize_seeds()
    device, _ = initialize_torch()
    evaluate_test_dataset(device, exp, 'facades')
    evaluate_test_dataset(device, exp, 'facades+fake_B')
    evaluate_test_dataset(device, exp, 'facades+rec_B')
    evaluate_test_dataset(device, exp, 'facades+fake_b+rec_B')
    evaluate_test_dataset(device, exp, 'facade_aug')
    evaluate_test_dataset(device, exp, 'facade+fake_B_aug')
    
def load_data(path, type, N=5):
    lines = []
    for i in range(N):
        file_path = os.path.join(path, f'{type}_{i}.txt')
        file = open(file_path, 'r')
        for line in file.readlines():
            tmp = [float(v) for v in line.split(',')]
            #tmp = [0]*(50-len(tmp))+tmp
            lines.append(tmp)
    return np.array(lines)

def errorbar(data, label, axis=0):
    plt.errorbar(np.arange(data.shape[1-axis]),
                 data.mean(axis=axis), yerr=data.std(axis=axis), label=label)
    plt.legend()

def boxplot(data, labels):
    plt.boxplot(data, showmeans=True, labels=labels)

def plotexp(title, facade, fakeb, recb, frec):
    plt.figure(figsize=(15,15))
    errorbar(facade, label='facades')
    errorbar(fakeb, label='fake_B')
    errorbar(recb, label='rec_B')
    errorbar(frec, label='fake_B+rec_B')
    plt.xlabel('Epoch')
    plt.ylabel('mIoU')
    plt.title(title)
    plt.show()
    plt.figure(figsize=(15,15))
    boxplot([facade[:,-1], fakeb[:,-1], recb[:,-1], frec[:,-1]],
            labels=['facades', 'fake_B', 'rec_B', 'fake_B+rec_B'])
    plt.ylabel('mIoU')
    plt.title(title)
    plt.show()

    print(np.round(np.mean([facade[:,-1], fakeb[:,-1], recb[:,-1], frec[:,-1]], 
                           axis=1), 3))
    print(np.round(np.std([facade[:,-1], fakeb[:,-1], recb[:,-1], frec[:,-1]], 
                          axis=1), 3))

    print('Fake_B/facades = ',
          np.round(np.mean(fakeb[:,-1])/np.mean(facade[:,-1]), 3))
    print('Rec_B/facades = ',
          np.round(np.mean(recb[:,-1])/np.mean(facade[:,-1]), 3))
    print('fake_B+rec_B/facades = ',
          np.round(np.mean(frec[:,-1])/np.mean(facade[:,-1]), 3))
    
    return np.mean([facade[:,-1], fakeb[:,-1], recb[:,-1], frec[:,-1]], axis=1)

def plotbest(data):
    best = 0
    for i in range(1, data.shape[0]):
        if data[i][-1] > data[i-1][-1]: best = i
    plt.plot(data[best])

def load_data_and_plot(exp):
    path = os.path.join(opt.dataroot,
                   'facades',
                   'experiments',
                   exp)
    facade = load_data(path, 'miou')
    path = os.path.join(opt.dataroot,
                   'facades+fake_B',
                   'experiments',
                   exp)
    fakeb = load_data(path, 'miou')
    path = os.path.join(opt.dataroot,
                   'facades+rec_B',
                   'experiments',
                   exp)
    recb = load_data(path, 'miou')
    path = os.path.join(opt.dataroot,
                   'facades+fake_B+rec_B',
                   'experiments',
                   exp)
    frec = load_data(path, 'miou')
    plotexp(exp, facade, fakeb, recb, frec)

def load_data_and_plot_earlystop(exp):
    path = os.path.join(opt.dataroot,
                   'facades',
                   'experiments',
                   exp)
    facade = load_data(path, 'miou')

    path = os.path.join(opt.dataroot,
                   'facades+fake_B',
                   'experiments',
                   exp)
    fakeb = load_data(path, 'miou')

    path = os.path.join(opt.dataroot,
                   'facades+rec_B',
                   'experiments',
                   exp)
    recb = load_data(path, 'miou')

    path = os.path.join(opt.dataroot,
                   'facades+fake_B+rec_B',
                   'experiments',
                   exp)
    rfb = load_data(path, 'miou')
    
    path = os.path.join(opt.dataroot,
                   'facades_aug',
                   'experiments',
                   exp)
    facade_aug = load_data(path, 'miou')

    path = os.path.join(opt.dataroot,
                   'facades+fake_B_aug',
                   'experiments',
                   exp)
    fakeb_aug = load_data(path, 'miou')
    
    plt.figure(figsize=(15,15))
    plotbest(facade)
    plotbest(fakeb)
    plotbest(recb)
    plotbest(rfb)
    plotbest(facade_aug)
    plotbest(fakeb_aug)
    plt.legend(['facades', 'fake_b', 'rec_b', 
                'rec_b+fake_b', 'facade (aug)', 'fake_b (aug)'])
    plt.xlabel('Epoch')
    plt.ylabel('mIoU')
    plt.title(exp)
    plt.show()
    
    
    # print(np.round(np.mean([facade[:,-1], fakeb[:,-1], recb[:,-1], rfb[:,-1], 
    #                        facade_aug[:,-1], fakeb_aug[:,-1]], axis=1), 3))
    # print(np.round(np.std([facade[:,-1], fakeb[:,-1], recb[:,-1], rfb[:,-1], 
    #                        facade_aug[:,-1], fakeb_aug[:,-1]], axis=1), 3))

    # print('Fake_B/facades = ',
    #       np.round(np.mean(fakeb[:,-1])/np.mean(facade[:,-1]), 3))
    # print('Rec_B/facades = ',
    #       np.round(np.mean(recb[:,-1])/np.mean(facade[:,-1]), 3))
    # print('fake_B+rec_B/facades = ',
    #       np.round(np.mean(rfb[:,-1])/np.mean(facade[:,-1]), 3))
    # print('facade (aug)/facades = ',
    #       np.round(np.mean(facade_aug[:,-1])/np.mean(facade[:,-1]), 3))
    # print('fake_b (aug)/facades = ',
    #       np.round(np.mean(fakeb_aug[:,-1])/np.mean(facade[:,-1]), 3))

if __name__ == "__main__":
    load_data_and_plot('ResNet50,lr=0.001,epochs=50')
    load_data_and_plot('ResNet101,lr=0.001,epochs=50')
    load_data_and_plot('ResNet101,lr=0.0002,epochs=100')
    load_data_and_plot('ResNet101,lr=0.0002,epochs=50')
    load_data_and_plot('ResNet50,lr=0.0002,epochs=50')
    load_data_and_plot_earlystop('ResNet101,lr=0.0002,epochs=50,earlystop')
    evaluate_test('ResNet101,lr=0.0002,epochs=50,earlystop')