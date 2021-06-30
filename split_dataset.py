# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 14:35:36 2020

Create the Semantic Segmentation dataset

@author: Renato B. Arantes
"""

import os
import sys
import glob
import argparse
import numpy as np

from PIL import Image
from random import shuffle

parser = argparse.ArgumentParser()
parser.add_argument('--imgroot', required=True, help='source image path')
parser.add_argument('--lblroot', required=True, help='source label path')
parser.add_argument('--dstroot',
                    required=True,
                    help='dataset output path')
parser.add_argument('--split',
                    type=int,
                    default=80,
                    required=False, help='Train split, default 80%')
parser.add_argument('--num_labels', type=int, required=True,
                    help='Dataset label number')
parser.add_argument('--dataset_size', type=int, default=0,
                    help='use N images/labels to create the dataset')
parser.add_argument('--new_size', nargs='+', type=int, required=False,
                    default=[0, 0],
                    help='New image/label size. Ex. --resize 256 256')
parser.add_argument('--resize', default='--new_size' in sys.argv,
                    help=argparse.SUPPRESS)
opt = parser.parse_args()


NEW_SIZE = tuple(opt.new_size)
NUM_LABELS = opt.num_labels
LABELS = set([i for i in range(NUM_LABELS)])


def copy_file(file_name, set_type, resize):
    lbl = Image.open(os.path.join(opt.lblroot, f'{file_name}.png'))
    if resize: lbl = lbl.resize(NEW_SIZE)
    lbl.save(os.path.join(opt.dstroot, set_type, f'{file_name}.png'))
    assert set(np.unique(lbl)).issubset(LABELS), \
        print(LABELS, set(np.unique(lbl)))

    img = Image.open(os.path.join(opt.imgroot, f'{file_name}.jpg'))
    if resize: img = img.resize(NEW_SIZE)
    img.save(os.path.join(opt.dstroot, set_type, f'{file_name}.jpg'))


def create_if_not_exist(path):
    if not os.access(path, os.F_OK):
        os.makedirs(path)


def create_set(files, set_type, start, stop):
    print(set_type, start, stop)
    for i in range(start, stop):
        file_name = os.path.splitext(os.path.basename(files[i]))[0]
        copy_file(file_name, set_type, opt.resize)


if __name__ == '__main__':
    create_if_not_exist(os.path.join(opt.dstroot, 'train'))
    create_if_not_exist(os.path.join(opt.dstroot, 'test'))

    files = set([f for f in glob.glob(os.path.join(opt.imgroot, '*.jpg'))])
    files = list(files)
    shuffle(files)
    if opt.dataset_size > 0:
        files = files[:opt.dataset_size]
        assert len(files) == opt.dataset_size

    train_split = int(len(files)*(opt.split/100.))
    print(f'train size = {train_split}, test size = {len(files)-train_split}')

    create_set(files, 'train', 0, train_split)
    create_set(files, 'test', train_split, len(files))
