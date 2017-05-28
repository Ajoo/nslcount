#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 23:04:04 2017

@author: ajoo
"""
import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable
from torch import nn
import torch.nn.parallel

from torchvision import transforms, datasets, models


import sys
import os

RESUME = False
PARALLEL = True
CUDA = True

BATCH_SIZE = 64
NUM_WORKERS = 4
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0
MOMENTUM = 0.9

START_EPOCH = 0
N_EPOCHS = 100

OUTPUT_FREQ = 10

class SeaLionVGG(models.VGG):
    def __init__(self, model, num_classes=6):
        super(models.VGG, self).__init__()
        
        if PARALLEL:
            self.features = nn.DataParallel(model().features)
        else:
            self.features = model().features
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2 * 2, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, num_classes),
        )
        self._initialize_weights()
        
def accuracy(outputs, targets):
    _, pred = outputs.data.topk(1, 1, True, True)
    return pred.eq(targets.data).sum()/pred.size(0)
        
class AverageMeter(object):
    def __init__(self):
        self.reset()
        
    def reset(self):
        try:
            self.history.append(self.avg)
        except:
            self.history = []
        self.sum = 0
        self.avg = 0
        self.count = 0
        
    def update(self, val, n=1):
        self.sum += val*n
        self.count += n
        self.avg = self.sum/self.count
        
    @property
    def serialized(self):
        pass
        
class Report(object):
    def __init__(self, *names):
        self.metrics = [AverageMeter() for n in range(len(names))]
        self.names = names

    def reset(self):
        for m in self.metrics:
            m.reset()
        
    def update(self, *metrics, n=1):
        for m, val in zip(self.metrics, metrics):
            m.update(val, n)
       
    @property
    def serialized(self):
        return self
        
    @serialized.setter
    def serialized(self, value):
        self.metrics = value.metrics
        self.names = value.names
            
    def __str__(self):
        return ', '.join('{} : {}'.format(n, v.avg) for n, v in zip(self.names, self.metrics))
        
#%% Load models
CHECKPOINT_FILE = os.path.join('..', 'Checkpoints', 'checkpoint_ADAM_{epoch}.pth.tar')

model = SeaLionVGG(models.vgg13)

loss_function = nn.CrossEntropyLoss()

#optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE,
#                            momentum=MOMENTUM, nesterov=False,
#                            weight_decay=WEIGHT_DECAY)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,
                            weight_decay=WEIGHT_DECAY)

if CUDA:
    model.cuda()
    loss_function.cuda()
    
# Resume?
try:
    if not RESUME:
        raise Exception('Resume turned off')
    checkpoint = torch.load(CHECKPOINT_FILE)
    
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optim_state'])
    train_report = checkpoint['train_report']
    val_report = checkpoint['val_report']
except:
    print("No checkpoint file found. Starting over!")
    train_report = Report('loss', 'accuracy')
    val_report = Report('loss', 'accuracy')

#%% Data Loaders
train_dir = os.path.join('..', 'Tiles', 'Train')
val_dir = os.path.join('..', 'Tiles', 'Val')

mean = [0.1578, 0.1614, 0.2701] #[0.485, 0.456, 0.406]
std = [0.8393, 0.7114, 0.6358] #[0.229, 0.224, 0.225]
trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
    ])

train_folder = datasets.ImageFolder(train_dir, trans)
val_folder = datasets.ImageFolder(val_dir, trans)

train_loader = torch.utils.data.DataLoader(train_folder, batch_size=BATCH_SIZE,
                                           num_workers=NUM_WORKERS, shuffle=True, pin_memory=CUDA)
val_loader = torch.utils.data.DataLoader(val_folder, batch_size=BATCH_SIZE,
                                           num_workers=NUM_WORKERS, shuffle=True, pin_memory=CUDA)
 
#%%    

def process_batches(loader, report, train=True):
    if train:
        model.train()
    else:
        model.eval()
    
    loader = iter(loader)
    
    for i, (tiles, targets) in enumerate(loader):
        if CUDA:
            targets = targets.cuda()
        tiles, targets = Variable(tiles), Variable(targets)
        
        
        outputs = model(tiles)
        loss = loss_function(outputs, targets)
        acc = accuracy(outputs, targets)
        
        report.update(loss.data[0], acc, tiles.size(0))
        
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        if i%OUTPUT_FREQ == 0:
            print('\tBatch {b}: {r}'.format(b=i, r=str(report)))
            
    report.reset()
        
        


for epoch in range(START_EPOCH, N_EPOCHS):
    print('Epoch {e}:'.format(e=epoch))
    
    #TRAIN
    print('Training...')
    process_batches(train_loader, train_report)
    
    #VAL
    print('Validating...')
    process_batches(val_loader, val_report, train=False)
    
    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optim_state': optimizer.state_dict(),
        'train_report': train_report,
        'val_report': val_report
                }, CHECKPOINT_FILE.format(epoch=epoch//OUTPUT_FREQ))
