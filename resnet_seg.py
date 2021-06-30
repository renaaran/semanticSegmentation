"""Created on Thu Mar  5 14:19:06 2020.

@author: Renato B. Arantes
"""
import os
import time
import torch
import importlib
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from utils import initialize_seeds, initialize_torch, calc_iou
from config import opt

from torch.nn.parallel.data_parallel import DataParallel

class CallbackContext(object):
    pass

def execute_replication_callbacks(modules):
    """
    Execute an replication callback `__data_parallel_replicate__` on each module created by original replication.

    The callback will be invoked with arguments `__data_parallel_replicate__(ctx, copy_id)`

    Note that, as all modules are isomorphism, we assign each sub-module with a context
    (shared among multiple copies of this module on different devices).
    Through this context, different copies can share some information.

    We guarantee that the callback on the master copy (the first copy) will be called ahead of calling the callback
    of any slave copies.
    """
    master_copy = modules[0]
    nr_modules = len(list(master_copy.modules()))
    ctxs = [CallbackContext() for _ in range(nr_modules)]

    for i, module in enumerate(modules):
        for j, m in enumerate(module.modules()):
            if hasattr(m, '__data_parallel_replicate__'):
                m.__data_parallel_replicate__(ctxs[j], i)

class DataParallelWithCallback(DataParallel):
    """
    Data Parallel with a replication callback.

    An replication callback `__data_parallel_replicate__` of each module will be invoked after being created by
    original `replicate` function.
    The callback will be invoked with arguments `__data_parallel_replicate__(ctx, copy_id)`

    Examples:
        > sync_bn = SynchronizedBatchNorm1d(10, eps=1e-5, affine=False)
        > sync_bn = DataParallelWithCallback(sync_bn, device_ids=[0, 1])
        # sync_bn.__data_parallel_replicate__ will be invoked.
    """

    def replicate(self, module, device_ids):
        modules = super(DataParallelWithCallback, self).replicate(module, device_ids)
        execute_replication_callbacks(modules)
        return modules


def load_model(model_name):
    model_lib = importlib.import_module('torchvision.models.segmentation')
    for name, cls in model_lib.__dict__.items():
        if name.lower() == model_name.lower():
            model_class = cls

    if model_class is None:
        raise ValueError(f"{model_name} not found!")

    print('**** Model:', model_name)
    model = model_class(num_classes=opt.num_labels, pretrained=False)
    model = model.to(device)
    return model


def create_dataset(dataset_name):
    global train_loader, val_loader
    dataset_lib = importlib.import_module('dataset')
    for name, cls in dataset_lib.__dict__.items():
        if name.lower() == dataset_name.lower():
            dataset_class = cls

    if dataset_class is None:
        raise ValueError(f"{dataset_name} not found!")

    print('**** Dataset:', dataset_name)
    train_dataset = dataset_class(opt.dataroot, root='train',
                                  augment=opt.augment)
    print(f'train_dataset size={len(train_dataset)}')
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=16, num_workers=8)
    val_dataset = dataset_class(opt.dataroot, root='val',
                                augment=False)
    print(f'val_dataset size={len(val_dataset)}')
    val_loader = DataLoader(val_dataset, shuffle=True, batch_size=4, num_workers=8)


def initialise():
    global model, device, ngpu
    initialize_seeds(opt.seed)
    device, ngpu = initialize_torch(opt.cuda)
    model = load_model(opt.model_name)
    model = DataParallelWithCallback(model, device_ids=[0, 1, 2, 3])
    create_dataset(opt.dataset_name)


def save_file(file_name, data):
    text_file = open(os.path.join(opt.outputFolder, file_name), "w")
    text_file.write(str(data).strip('[]'))
    text_file.close()


def must_stop(metric):
    N = 10
    if not opt.earlystop or len(metric) < N:
        return False
    return True if len(set(np.round(metric[-N:], 2))) == 1 else False


def is_best_epoch(metric, total_epochs):
    N = 40
    if len(metric) < N:
        return False
    if len(metric) == N:
        return True
    lastN = sorted(metric[N-1:])
    return lastN[-1] == metric[-1]


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
            # print(X.shape, y.shape)
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
                # print(X.shape, y.shape, output.shape)
                loss = ce_loss(output, y)
                probs = torch.functional.F.softmax(output, 1)
                y_pred = torch.argmax(probs, dim=1)
                correct += torch.sum(y_pred == y).item()
                total += (y.shape[-1]*y.shape[-2]*y.shape[0])
                iou.append(calc_iou(y_pred, y))
                val.append(loss.item())
            val_loss.append(np.mean(val))
            miou.append(np.mean(iou))
            acc.append(correct/total)

        best = is_best_epoch(miou, epochs)
        if best:
            best_val_loss = val_loss[-1]
            best_epoch_loss = epoch_loss[-1]
            best_acc = acc[-1]
            best_miou = miou[-1]
            torch.save(model.state_dict(),
                       os.path.join(opt.outputFolder, f'model_{run}.pth'))

        print(i,
              "mIoU: ", round(miou[-1], 4),
              " - Accuracy: ", round(acc[-1], 4),
              " - Loss: ", round(val_loss[-1], 4), "*" if best else "")

        run_file.write(f'{i},{miou[-1]},{val_loss[-1]},{acc[-1]}\n')

        # early stopping
        if must_stop(miou):
            print('***** Early stopping! :)')
            break

    train_time += time.time() - start_time
    print(f"Train_time={train_time/60} (min)")

    # plt.subplots(figsize=(15, 8))
    # plt.imsave(os.path.join(opt.outputFolder, f"predicition_{run}.png"),
    #            np.hstack((y_pred.squeeze().cpu(), y.squeeze().cpu())))

    save_file(f'val_loss_{run}.txt', val_loss)
    save_file(f'epoch_loss_{run}.txt', epoch_loss)
    save_file(f'acurracy_{run}.txt', acc)
    save_file(f'miou_{run}.txt', miou)

    run_file.close()

    return best_val_loss, best_epoch_loss, best_acc, best_miou


if __name__ == "__main__":
    epoch_loss = []
    val_loss = []
    acc = []
    miou = []
    for i in range(3):
        print(i, '******************************')
        initialise()
        vl, el, a, m = train(i, opt.epochs)
        val_loss.append(vl)
        epoch_loss.append(el)
        acc.append(a)
        miou.append(m)

    print('val_loss:', val_loss,
          round(np.mean(val_loss), 3),
          round(np.std(val_loss), 3))
    print('epoch_loss:', epoch_loss,
          round(np.mean(epoch_loss), 3),
          round(np.std(epoch_loss), 3))
    print('accuracy:', acc,
          round(np.mean(acc), 3),
          round(np.std(acc), 3))
    print('miou:', miou,
          round(np.mean(miou), 3),
          round(np.std(miou), 3))
