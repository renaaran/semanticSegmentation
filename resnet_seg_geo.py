#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 14:19:06 2020

@author: Renato B. Arantes
"""
import os
import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from utils import initialize_seeds, initialize_torch, calc_iou
from dataset import GeoDataset
from config import opt

from torchvision.models.segmentation import deeplabv3_resnet50, \
    deeplabv3_resnet101

def load_model():
    global model
    if opt.model == 50:
        model = deeplabv3_resnet50(num_classes=GeoDataset.NUM_LABELS,
                                   pretrained=False)
    elif opt.model == 101:
        model = deeplabv3_resnet101(num_classes=GeoDataset.NUM_LABELS,
                                    pretrained=False)

    else: raise Exception(f'Invalid model: {opt.model}')
    model = model.to(device)

def create_dataset():
    global train_loader, val_loader
    train_dataset = GeoDataset(opt.dataroot, root='train')
    print(f'train_dataset size={len(train_dataset)}')
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=2)
    val_dataset = GeoDataset(opt.dataroot, root='val')
    print(f'val_dataset size={len(val_dataset)}')
    val_loader = DataLoader(val_dataset, shuffle=True, batch_size=1)

def initialise():
    global device, ngpu
    initialize_seeds(opt.seed)
    device, ngpu = initialize_torch(opt.cuda)
    load_model()
    create_dataset()

def save_file(file_name, data):
    text_file = open(os.path.join(opt.outputFolder, file_name), "w")
    text_file.write(str(data).strip('[]'))
    text_file.close()

def must_stop(metric):
    N = 10
    if not opt.earlystop or len(metric) < N: return False
    return True if len(set(np.round(metric[-N:], 2))) == 1 else False

def train(run, epochs):

    run_file = open(os.path.join(opt.outputFolder, f"run_{run}.csv"), "w")

    epoch_loss = []
    val_loss = []
    acc = []
    miou = []
    train_time = 0
    ce_loss = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    start_time = time.time()

    for i in range(epochs):
        ep = 0
        model.train()
        for X, y in train_loader:
            optim.zero_grad()
            X = X.to(device)
            y = y.to(device)
            output = model(X)['out']
            loss = ce_loss(output, y)
            loss.backward()
            ep += loss.item()
            optim.step()
        epoch_loss.append(ep)

        correct = 0
        total = 0
        val = []
        iou = []
        model.eval()
        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(device)
                y = y.to(device)
                output = model(X)['out']
                loss = ce_loss(output, y)
                probs = torch.functional.F.softmax(output, 1)
                y_pred = torch.argmax(probs, dim=1)
                correct += torch.sum(y_pred == y).item()
                total += (y.shape[-1]*y.shape[-2])
                iou.append(calc_iou(y_pred, y))
                val.append(loss.item())
            val_loss.append(np.mean(val))
            miou.append(np.mean(iou))
            acc.append(correct/total)

        print(i,
              "mIoU: ", round(miou[-1], 2),
              " - Accuracy: ", round(acc[-1], 2),
              " - Loss: ", round(val_loss[-1], 2))

        run_file.write(f'{i},{miou[-1]},{val_loss[-1]},{acc[-1]}\n')

        # early stopping
        if must_stop(miou):
            print('***** Early stopping! :)')
            break

    train_time += time.time() - start_time
    print(f"Train_time={train_time/60} (min)")

    plt.subplots(figsize=(15, 8))
    plt.imsave(os.path.join(opt.outputFolder, f"predicition_{run}.png"),
               np.hstack((y_pred.squeeze().cpu(), y.squeeze().cpu())))

    save_file(f'val_loss_{run}.txt', val_loss)
    save_file(f'epoch_loss_{run}.txt', epoch_loss)
    save_file(f'acurracy_{run}.txt', acc)
    save_file(f'miou_{run}.txt', miou)

    run_file.close()

    torch.save(model.state_dict(),
               os.path.join(opt.outputFolder, f'model_{run}.pth'))

    return val_loss[-1], epoch_loss[-1], acc[-1], miou[-1]

if __name__ == "__main__":
    epoch_loss = []
    val_loss = []
    acc = []
    miou = []
    for i in range(5):
        print(i, '******************************')
        initialise()
        vl, el, a, m = train(i, opt.epochs)
        val_loss.append(vl)
        epoch_loss.append(el)
        acc.append(a)
        miou.append(m)

    print('val_loss:', val_loss, round(np.mean(val_loss), 3),
                                 round(np.std(val_loss), 3))
    print('epoch_loss:', epoch_loss, round(np.mean(epoch_loss), 3),
                                     round(np.std(epoch_loss), 3))
    print('accuracy:', acc, round(np.mean(acc), 3),
                            round(np.std(acc), 3))
    print('miou:', miou, round(np.mean(miou), 3),
                         round(np.std(miou), 3))
