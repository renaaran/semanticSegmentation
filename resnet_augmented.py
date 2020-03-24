#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 18:42:13 2020

@author: Renato B. Arantes
"""

import torch
import torch.nn as nn

from torchvision.datasets import CIFAR10
from torchvision import transforms
from torchvision.transforms import Compose

import matplotlib.pyplot as plt
import time

from models.resnet import resnet18


device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

print(device)

######################### 1 - load dataset
train_transform = Compose([
    # transforms.RandomHorizontalFlip(p=0.5),
    # transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize([0, 0, 0], [1, 1, 1])
])

test_transform = Compose([
    transforms.ToTensor(),
    transforms.Normalize([0, 0, 0], [1, 1, 1])
])

cifar10_train = CIFAR10(root = "./datasets", train=True, download = True, transform=train_transform)
train_loader = torch.utils.data.DataLoader(cifar10_train, batch_size=128, shuffle=True)
cifar10_test = CIFAR10(root = "./datasets", train=False, download = True, transform=test_transform)
test_loader = torch.utils.data.DataLoader(cifar10_test, batch_size=128, shuffle=True)

######################### 2 - Set up training
model = resnet18(3, 10)
model = model.to(device)
loss_fn = nn.CrossEntropyLoss()
LR = 0.001
optim = torch.optim.Adam(model.parameters(), lr = LR, weight_decay=0.0001)
EPOCHS = 50
epoch_loss = []
val_loss = []
acc = []
train_time = 0

######################### 3 - Train and evaluate
for i in range(EPOCHS):
    start_time = time.time()
    ep = 0
    model.train()
    for X_b, y_b in train_loader:
        optim.zero_grad()
        X_b = X_b.to(device)
        y_b = y_b.to(device)
        output = model(X_b)
        print(output.shape, y_b.shape)
        loss = loss_fn(output, y_b)
        loss.backward()
        ep += loss.item()
        optim.step()
    epoch_loss.append(ep)

    correct = 0
    total = 0
    val = 0
    model.eval()
    for X_b, y_b in test_loader:
        X_b = X_b.to(device)
        y_b = y_b.to(device)
        output = model(X_b)
        loss = loss_fn(output, y_b)
        val += loss.item()
        probs = torch.functional.F.softmax(output, 1)
        label = torch.argmax(probs, dim=1)
        correct += torch.sum(label == y_b).item()
        total += y_b.shape[0]
    val_loss.append(val)
    acc.append(round(correct/total,2))

    print(i, " - Accuracy: ", round(correct/10000,2), "Loss: ", round(val,1))

train_time += time.time() - start_time
print(f"Train_time={train_time} (min)")

# 5 - Plot
fig, ax = plt.subplots(figsize=(15, 8))
plt.plot(range(EPOCHS), epoch_loss , color='r')
plt.plot(range(EPOCHS), val_loss, color='b')
plt.legend(["Train Loss", "Validation Loss"])
plt.xlabel("Epochs")
plt.ylabel("Loss")
ax.grid(True)
fig, ax = plt.subplots(figsize=(15, 8))
plt.plot(range(EPOCHS), acc , color='g')
plt.legend(["Validation Accuracy"])
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("No Augmentation")
ax.grid(True)