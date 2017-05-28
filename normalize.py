#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 21 12:40:47 2017

@author: ajoo
"""

import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable
from torch import nn

from torchvision import transforms, datasets, models

from itertools import chain

import sys
import os

BATCH_SIZE = 128
NUM_WORKERS = 4

#%% Data Loaders
train_dir = os.path.join('..', 'Tiles', 'Train')
val_dir = os.path.join('..', 'Tiles', 'Val')

trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

train_folder = datasets.ImageFolder(train_dir, trans)
val_folder = datasets.ImageFolder(val_dir, trans)


train_loader = torch.utils.data.DataLoader(train_folder, batch_size=BATCH_SIZE,
                                           num_workers=NUM_WORKERS, shuffle=True, pin_memory=False)
val_loader = torch.utils.data.DataLoader(val_folder, batch_size=BATCH_SIZE,
                                           num_workers=NUM_WORKERS, shuffle=True, pin_memory=False)

mean = torch.zeros(3)
std = torch.zeros(3)
N = 0
#%%
for tiles, targets in chain(iter(train_loader),iter(val_loader)):
    N += tiles.size(0)
    for i in range(3):
        mean[i] += torch.sum(tiles[:,i])/64**2
        std[i] += torch.sum(tiles[:,i]**2)/64**2
        
mean /= N
std /= N
std -= mean**2