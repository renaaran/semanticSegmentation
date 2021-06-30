# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 18:12:37 2020

@author: Renato B. Arantes
"""
import re
import os
import glob
import numpy as np
import argparse
import matplotlib.pyplot as plt

from albumentations import (
    HorizontalFlip,
    VerticalFlip,
    Transpose,
    RandomRotate90
)

from PIL import Image

NUM_LABELS = 12

AUGMENTS = [RandomRotate90(p=1),
            HorizontalFlip(p=1),
            VerticalFlip(p=1),
            Transpose(p=1)]
    
parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', required=True, 
                    help='expriments results path')
opt = parser.parse_args()

def read(index, ext):
    file_name = f'{index}.{ext}'
    img_path = os.path.join(opt.dataroot, 'train', file_name)
    return Image.open(img_path)

def plot(img, lbl):
    _,  ax = plt.subplots(1,2)
    ax[0].imshow(img)
    ax[1].imshow(lbl)
    plt.show()    
    
def augment():
    regex = re.compile(r'\d+')
    dir = glob.glob(os.path.join(opt.dataroot, 'train', '*.jpg'))
    cnt = len(dir)+1
    print(cnt)
    for file_name in sorted(dir):
        print(file_name)
        idx = int(regex.findall(os.path.basename(file_name))[0])
        img = np.array(read(idx, 'jpg'))
        lbl = read(idx, 'png')
        for i in range(len(AUGMENTS)):
            aug = AUGMENTS[i](image=img, mask=np.array(lbl))
            aug_img, aug_lbl = aug['image'], aug['mask']
            plt.imsave(os.path.join(opt.dataroot, 'train', f'{cnt}.jpg'), aug_img)
            lbl_indexed = Image.fromarray(aug_lbl).convert('P')
            lbl_indexed.putpalette(lbl.getpalette())
            lbl_indexed.save(os.path.join(opt.dataroot, 'train', f'{cnt}.png'))
            cnt += 1
            
            assert len(lbl.getcolors()) == len(lbl_indexed.getcolors())
            
if __name__ == '__main__':
    augment()          
