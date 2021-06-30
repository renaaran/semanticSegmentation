#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 16:20:34 2020

@author: Renato Barros Arantes

Search over all experiments directory to sumarize the results.
"""
import os
import glob
import numpy as np

def get_value(file_path):
    N = 40
    l = []
    with open(file_path) as txt:
        for row in txt:
            l = [float(v) for v in row.split(',')]
    return float(sorted(l[N:])[-1])

def get_model(file_path):
    with open(file_path) as txt:
        for row in txt:
            if 'fcn_resnet101' in row:
                return 'fcn'
            elif 'DeeplabV3_resnet101' in row:
                return 'DeeplabV3'
    raise ValueError("Can't define model!")

if __name__ == '__main__':
    summary = {}
    exp_model = {}
    for data_path in sorted(glob.iglob('datasets/**/*.txt', recursive=True)):
        path, file = os.path.split(data_path)
        path_elements = path.split(os.path.sep)
        dataset, exp_name = path_elements[1], path_elements[3]
        metric = file.split('_')[0]
        if 'options.txt' in data_path:
            exp_model[exp_name] = get_model(data_path)
            continue
        if metric in {'epoch', 'val'}:
            continue
        key = (dataset, exp_name, metric)
        value = get_value(data_path)
        if key not in summary:
            summary[key] = [value]
        else:
            summary[key].append(value)

    for key, values in summary.items():
        print((*key, exp_model[key[1]]), round(np.mean(values), 3), round(np.std(values), 3))
