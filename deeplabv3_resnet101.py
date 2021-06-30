#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 14:02:45 2019

@author: Renato B. Arantes
"""
import os
import time
import torch
import torch.nn as nn
import numpy as np
import argparse
import datetime
import matplotlib.pyplot as plt

import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms

from torchvision.models.segmentation import deeplabv3_resnet101
from torch.utils import data

from PIL import Image
from utils import initializeSeeds, initializeTorch

IGNORE_LABEL = 255
NUM_LABELS = 12
SEED = 999

parser = argparse.ArgumentParser()
parser.add_argument('--inputPath', required=False, help='Input path.')
parser.add_argument('--outputPath', required=False, help='Output path.')
parser.add_argument('--epochs', required=False, type=int, default=500, help='Number of epochs for training.')
parser.add_argument('--batchSize', required=False, type=int, default=16, help='Batch size.')
opt = parser.parse_args()

class FacadeDataset(data.Dataset):
    def __init__(self, split):
        self.split = split

    def _set_files(self):
        if self.split in ["train", "val"]:
            fileListPath = os.path.join(opt.inputPath, "index", self.split + ".txt")
            fileList = tuple(open(fileListPath, "r"))
            fileList = [id_.rstrip() for id_ in fileList]
            self.files = fileList
        else:
            raise ValueError("Invalid split name: {}".format(self.split))

    def _load_data(self, index):
        # Set paths
        imageId = self.files[index]
        imagePath = os.path.join(opt.inputPath, "image", imageId + ".png")
        labelPath = os.path.join(opt.inputPath, "mask", imageId + ".png")
        # Load an image
        image = plt.imread(imagePath)
        label = np.array(Image.open(labelPath), dtype=np.int32)
        return imageId, image, label

def createDataset():
    global dataloader
    # Create the dataset
    dataset = dset.ImageFolder(root=opt.inputPath,
                               transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225]),
                               ]))
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                             shuffle=True, num_workers=4)

def loadModel():
    global model
    # model = torch.hub.load('pytorch/vision:v0.5.0', 'deeplabv3_resnet101',
    #                         num_classes=NUM_LABELS, pretrained=False, progress=True)
    model = deeplabv3_resnet101(num_classes=NUM_LABELS)
    model.train()
    print(model)
    return model

def initialise():
    global device, ngpu
    loadModel()
    initializeSeeds(SEED)
    device, ngpu = initializeTorch()
    createDataset()

def printTime(t0=None):
    now = datetime.datetime.now()
    print(now.strftime("%d/%m/%Y %H:%M:%S"))
    if t0 != None:
        print('Execution time: %.4f (secs)' % (time.time()-t0))

def train():
    losses = []
    # Loss definition
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL)
    criterion.to(device)
    # The optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999))
    # Training Loop
    printTime()
    print("Starting training loop...")
    trainStartTime = t0 = time.time()
    # For each epoch
    for epoch in range(opt.epochs):
        # For each batch in the dataloader
        for i, images_, labels_ in enumerate(dataloader, 0):
            optimizer.zero_grad()
            images = images_.to(device)
            labels = labels_.to(device)
            # Forward pass
            logits = model(images)
            print(logits.shape, labels.shape)
            # Calculate loss
            iterLoss = 0
            for logit in logits:
                # Resize labels for {100%, 75%, 50%, Max} logits
                _, _, H, W = logit.shape
                print(H, W)
                iterLoss += criterion(logit, labels.to(device))
            # Propagate backward
            iterLoss /= len(logits)
            iterLoss.backward()
        # Update weights with accumulated gradients
        optimizer.step()
        # Save Losses for plotting later
        losses.add(iterLoss.item())
        # Output training stats
        print('[%d/%d][%d/%d]\tLoss: %.4f - time %.4f'
                % (epoch+1, opt.epochs, i, len(dataloader), losses[-1], time.time()-t0))
        t0 = time.time()

    printTime(trainStartTime)

def main():
    initialise()
    train()

if __name__ == "__main__":
    from torchsummary import summary
    model = loadModel()
    summary(model.cuda(), (3, 224, 224))
    input_image = np.random.randn(64, 64, 3)
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(input_image)
    # create a mini-batch as expected by the model
    input_batch = input_tensor.unsqueeze(0).float()

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    model.eval()
    output = model(input_batch)['out'][0]
    print(output.shape)
    output_predictions = output.argmax(0)
    print(output_predictions.shape)
    print(output_predictions)