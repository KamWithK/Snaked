#!/usr/bin/env python
# coding: utf-8
import os, time, torch

import numpy as np
import matplotlib.pyplot as plt

from torchvision import models, transforms
from torch import nn, optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch_lr_finder import LRFinder
from Models.Trainer import Trainer
from Data.SnakeDataset import SnakeDataset

data = {
    "train": ["../train", "../train_labels.csv", 0, 0.95],
    "validation": ["../train", "../train_labels.csv", 0.95, 1],
#    "test": ["../train", "../train_labels.csv", 0.975, 1]
}

image_transforms = {
    "train":
    transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1)),
        transforms.RandomRotation(90),
        transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
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

def train_test(trainer, path):
    if not os.path.exists("Saved"):
        os.mkdir("Saved")
    if not os.path.exists(path):
        os.mkdir(path)
    trainer.train()
    #trainer.evaluate()

def find_lr(model, optimizer, criterion, save_folder):
    lr_finder = LRFinder(model, optimizer, criterion, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    lr_finder.range_test(data_loaders["train"], end_lr=10, num_iter=1000)
    lr_finder.plot()
    plt.savefig(save_folder + "/LRvsLoss.png")
    plt.close()

# Pass in a dictionary for data_map like the following (note that for full datasets positions DON'T have to be included):
#data_map = {
    #"train": [path_to_data, path_to_csv, position_start, position_end],
    #"validation": [path_to_data, path_to_csv, position_start, position_end],
    #"test": [path_to_data, path_to_csv, position_start, position_end]
#}
def get_loaders(data_map, transforms=transforms.ToTensor(), shuffle=True, batch_size=128, num_workers=5):
        data_loaders = {}

        for name in data_map:
            path_to_data = data_map[name][0]
            path_to_csv = data_map[name][1]
            data = SnakeDataset(path_to_data, path_to_csv, transforms[name])
            data_loaders[name] = DataLoader(data, batch_size=batch_size, num_workers=num_workers),
            if len(data_map[name]) == 4:
                num_images, indicies = len(data), len(data)
                if shuffle == True:
                    indicies = np.random.permutation(num_images)
                position = [int(data_map[name][2] * num_images), int(data_map[name][3] * num_images)]
                sampler = SubsetRandomSampler(indicies[position[0]: position[1]])
                data_loaders[name] = DataLoader(data, sampler=sampler, batch_size=batch_size, num_workers=num_workers)
        return data_loaders

data_loaders = get_loaders(data, image_transforms)
# Criterion
criterion = nn.CrossEntropyLoss()

# MobileNet V2 model
print("Training MobileNet V2 model - " + str(time.strftime("%Y-%m-%d %H:%M:%S")))
model = models.mobilenet_v2(pretrained=True)
model.classifier[1] = nn.Sequential(
    nn.Linear(model.classifier[1].in_features, 1000),
    nn.ReLU6(),
    nn.Dropout(0.2),
    nn.Linear(1000, 85)
)
optimizer = optim.AdamW(model.parameters())
#find_lr(model, optimizer, criterion, "Saved/MobileNetV2 - Retrained")
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, 3e-3, epochs=100, steps_per_epoch=len(data_loaders["train"]))
trainer = Trainer(model, image_transforms, criterion, optimizer, scheduler, "Saved/MobileNetV2 - Retrained/Model.tar", data_loaders)
train_test(trainer, "Saved/MobileNetV2 - Retrained")

# SqueezeNet model
print("\nSqueezeNet 1_1 model - " + str(time.strftime("%Y-%m-%d %H:%M:%S")))
model = models.squeezenet1_1(pretrained=True)
model.classifier[1] = nn.Conv2d(512, 85, (1, 1), (1, 1))
optimizer = optim.AdamW(model.parameters())
#find_lr(model, optimizer, criterion, "Saved/SqueezeNet - Retrained")
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, 3e-3, epochs=100, steps_per_epoch=len(data_loaders["train"]))
trainer = Trainer(model, image_transforms, criterion, optimizer, scheduler, "Saved/SqueezeNet - Retrained/Model.tar", data_loaders)
train_test(trainer, "Saved/SqueezeNet - Retrained")

# ResNet50 model
print("\nTraining ResNet152 model - " + str(time.strftime("%Y-%m-%d %H:%M:%S")))
model = models.resnet152(pretrained=True)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 1000),
    nn.ReLU6(),
    nn.Dropout(0.2),
    nn.Linear(1000, 85)
)
if torch.cuda.device_count() > 1:
            nn.DataParallel(model)
optimizer = optim.AdamW(model.parameters())
#find_lr(model, optimizer, criterion, "Saved/ResNet50 - Retrained")
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, 1e-2, epochs=100, steps_per_epoch=len(data_loaders["train"]))
trainer = Trainer(model, image_transforms, criterion, optimizer, scheduler, "Saved/ResNet50 - Retrained/Model.tar", data_loaders)
train_test(trainer, "Saved/ResNet50 - Retrained")
