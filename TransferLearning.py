#!/usr/bin/env python
# coding: utf-8

import torch, time

import numpy as np

from torch import nn, optim
from torchvision import transforms, models
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from SnakeDataset import SnakeDataset
from TrainHelper import LossAccuracyKeeper

transforms = {
    "train":
    transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1)),
        transforms.RandomRotation(15),
        transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224), #ImgNet standards
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), # ImgNet standards
    ]),
    "validation":
    transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224), #ImgNet standards
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), # ImgNet standards
    ])
}

def split_data(transforms=None):
    # Train - fit model, Validation - tune hyperparameters, Test - final results
    train_data = SnakeDataset("../train", "../train_labels.csv", transforms["train"])
    validation_data = SnakeDataset("../train", "../train_labels.csv", transforms["validation"])
    test_data = SnakeDataset("../train", "../train_labels.csv", transforms["validation"])

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

train_loader, validation_loader, test_loader = split_data(transforms=transforms)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.vgg16(pretrained=True)
model.to(device)

# Freeze all weights
for param in model.parameters():
    param.requires_grad = False

model.classifier[6] = nn.Sequential(
    nn.Linear(model.classifier[6].in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 85),
    nn.LogSoftmax(dim=1)
)

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters())