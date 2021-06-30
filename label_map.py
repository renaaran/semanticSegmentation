#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 14:39:48 2020

@author: Renato B. Arantes

Maps from cycleGAN fake_A to indexed using real_B and original
indexed image.
"""
import os
import glob
import numpy as np
from PIL import Image

from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000

color_map = dict()

IDX_PATH = '/home/CAMPUS/180178991/desenv/semanticSegmentation/datasets/facades_new/train'
CYC_PATH = '/home/CAMPUS/180178991/desenv/pytorch-CycleGAN-and-pix2pix/results/facades_cyclegan.AtoB/test_latest/images'
DST_PATH = '/home/CAMPUS/180178991/desenv/semanticSegmentation/datasets/facades_new+rec_B/train'

def color_distance(rgb1, rgb2):
    c = tuple(rgb1)+tuple(rgb2)
    if c in color_map:
        return color_map[c]
    else:
        color1_rgb = sRGBColor(rgb1[0], rgb1[1], rgb1[2]);
        color1_lab = convert_color(color1_rgb, LabColor);

        color2_rgb = sRGBColor(rgb2[0], rgb2[1], rgb2[2]);
        color2_lab = convert_color(color2_rgb, LabColor);

        d = delta_e_cie2000(color1_lab, color2_lab)
        color_map[c] = d
        return d

def color_to_index(real, indexed):
    '''
    creates a map from RGB colors to indexed
    '''
    index_map = dict()
    for i in range(real.size[0]):
        for j in range(real.size[1]):
            p = indexed.getpixel((i,j))
            c = tuple(real.getpixel((i,j)))
            if c not in index_map:
                index_map[c] = p
            elif index_map[c] != p:
                raise Exception(f'Invalid label {p} at ({c}) for {index_map[c]}')

    return index_map

if __name__ == "__main__":
    for file_path in sorted(glob.glob(os.path.join(IDX_PATH, '*.png'))):
        print(os.path.basename(file_path))
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        fake = np.array(Image.open(os.path.join(CYC_PATH, f'{file_name}_fake_A.png')))
        real = Image.open(os.path.join(CYC_PATH, f'{file_name}_real_B.png'))
        indexed = Image.open(file_path)

        index_map = color_to_index(real, indexed)

        real = np.array(real)
        real_unique = np.unique(real.reshape(-1, real.shape[2]), axis=0)
        fake_mapped = np.zeros(indexed.size, dtype=np.uint8)
        for i in range(fake.shape[0]):
            for j in range(fake.shape[1]):
                d = [color_distance(real_unique[k], fake[i,j])
                     for k in range(real_unique.shape[0])]
                c = real_unique[d.index(min(d))]
                fake_mapped[i,j] = index_map[tuple(c)]

        fake_indexed = Image.fromarray(fake_mapped).convert('P')
        fake_indexed.putpalette(indexed.getpalette())

        assert len(indexed.getcolors()) == len(fake_indexed.getcolors())

        fake_indexed.save(os.path.join(DST_PATH, f'{file_name}_fake.png'))
        recb = Image.open(os.path.join(CYC_PATH, f'{file_name}_rec_B.png'))
        recb.save(os.path.join(DST_PATH, f'{file_name}_fake.jpg'))