#!/usr/bin/env python
# coding: utf-8

import torch, torchvision
import torch.nn.functional as F
import os, io, time

import pandas as pd
import numpy as np
import matplotlib.pyplot as pyplot
import torch.optim as optim

from PIL import Image
from torch import nn
from torchvision import datasets, transforms, models
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from SnakeDataset import SnakeDataset

img_size = 128

data_transforms = transforms.Compose([
    transforms.Resize(img_size),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.RandomCrop(img_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

def split_data(transforms=None):
    # Train - fit model, Validation - tune hyperparameters, Test - final results
    train_data = SnakeDataset(img_dir="../train", csv_path="../train_labels.csv", transforms=transforms)
    validation_data = SnakeDataset(img_dir="../train", csv_path="../train_labels.csv", transforms=transforms)
    test_data = SnakeDataset(img_dir="../train", csv_path="../train_labels.csv", transforms=transforms)

    num_images = len(train_data)
    shuffle = np.random.permutation(num_images)

    # Position of where to start and end splitting data
    train_end = int(num_images * 0.95)
    validation_start = int(num_images * 0.95)
    validation_end = int(num_images * 0.9975)
    test_start = int(num_images * 0.975)

    train_sampler = SubsetRandomSampler(shuffle[:train_end])
    validation_sampler = SubsetRandomSampler(shuffle[validation_start:validation_end])
    test_sampler = SubsetRandomSampler(shuffle[test_start:])

    train_loader = DataLoader(train_data, sampler=train_sampler)
    validation_loader = DataLoader(validation_data, sampler=validation_sampler)
    test_loader = DataLoader(test_data, sampler=test_sampler)

    return train_loader, validation_loader, test_loader

train_loader, validation_loader, test_loader = split_data(transforms=data_transforms)
