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
from dataset import FacadeDataset, NUM_LABELS
from config import opt

from torchvision.models.segmentation import deeplabv3_resnet50, \
    deeplabv3_resnet101

def load_model():
    global model
    if opt.model == 50:
        model = deeplabv3_resnet50(num_classes=NUM_LABELS, pretrained=False)
    elif opt.model == 101:
        model = deeplabv3_resnet101(num_classes=NUM_LABELS, pretrained=False)
    else: raise Exception(f'Invalid model: {opt.model}')
    model = model.to(device)

def create_dataset():
    global train_loader, test_loader
    train_dataset = FacadeDataset(opt.dataroot, opt.augment, root='train')
    print(f'train_dataset size={len(train_dataset)}')
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=16)
    test_dataset = FacadeDataset(opt.dataroot, opt.augment, root='test')
    print(f'test_dataset size={len(test_dataset)}')
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=1)

def initialise():
    global device, ngpu, dtframe
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
    optim = torch.optim.Adam(model.parameters(), lr=opt.lr)
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
        val = 0
        iou = []
        model.eval()
        for X, y in test_loader:
            X = X.to(device)
            y = y.to(device)
            output = model(X)['out']
            loss = ce_loss(output, y)
            val += loss.item()
            probs = torch.functional.F.softmax(output, 1)
            y_pred = torch.argmax(probs, dim=1)
            correct += torch.sum(y_pred == y).item()
            total += (y.shape[-1]*y.shape[-2])
            iou.append(calc_iou(y_pred, y))
        val_loss.append(val)
        miou.append(np.mean(iou))
        acc.append(correct/total)

        print(i,
              "mIoU: ", round(miou[-1], 2),
              " - Accuracy: ", round(acc[-1], 2),
              " - Loss: ", round(val,1))

        run_file.write(f'{i},{round(miou[-1], 2)},{round(val,2)}\n')

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

def test_model_creat():
    SIZE = 224
    IMG_SIZE = (SIZE, SIZE)
    from torchsummary import summary
    import numpy as np
    import torchvision.transforms as transforms

    #model = SemanticResnet18(3, 12)
    model = deeplabv3_resnet50(num_classes=NUM_LABELS, pretrained=False)
    print(model)
    summary(model.cuda(), (3, *IMG_SIZE))
    input_images = np.random.randn(2, *IMG_SIZE, 3)
    input_tensor = torch.zeros((2, 3, *IMG_SIZE))

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
    ])

    for i in range(input_images.shape[0]):
        input_tensor[i] = preprocess(input_images[i])

    input_tensor = input_tensor.cuda()
    # create a mini-batch as expected by the model
    input_batch = input_tensor.float()
    model(input_batch)

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

    print('val_loss:', val_loss)
    print('epoch_loss:', epoch_loss)
    print('accuracy:', acc)
    print('miou:', miou)
