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
from torch.utils.data import Dataset, DataLoader

class SnakeDataset(Dataset):
    # Create/Initialize variables
    def __init__(self, img_dir, csv_path, transforms=None, test=False):
        self.df = pd.read_csv(csv_path, usecols=["scientific_name", "filename"])
        self.img_names = self.df.index.values
        self.csv_path = csv_path
        self.img_dir = img_dir
        self.transforms = transforms
    
    def __len__(self):
        return len(self.img_names)
    
    # Return a single pair (img_tensor, label)
    def __getitem__(self, index):
        img = Image.open(self.img_names(index))
        img_tensor = transforms.ToTensor(img)
        label = self.df.at(index, "scientific_name")
        return (img_tensor, label)
