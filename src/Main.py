#!/usr/bin/env python
# coding: utf-8
import os, time, torch

from torchvision import models, transforms
from torch import nn, optim

from Models.Trainer import Trainer

data = {
    "train": ["../train", "../train_labels.csv", 0, 0.95],
    "validation": ["../train", "../train_labels.csv", 0.975, 1],
#    "test": ["../train", "../train_labels.csv", 0.975, 1]
}

image_transforms = {
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

def train_test(trainer, path):
    if not os.path.exists("Saved"):
        os.mkdir("Saved")
    if not os.path.exists(path):
        os.mkdir(path)
    trainer.train(path)
    #trainer.evaluate(path)

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
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, 1, total_steps=20)
trainer = Trainer(model, image_transforms, criterion, optimizer, scheduler, "Saved/MobileNetV2 - Retrained/Model.tar", data)
train_test(trainer, "Saved/MobileNetV2 - Retrained")

# SqueezeNet model
print("\nSqueezeNet 1_1 model - " + str(time.strftime("%Y-%m-%d %H:%M:%S")))
model = models.squeezenet1_1(pretrained=True)
model.classifier[1] = nn.Conv2d(512, 85, (1, 1), (1, 1))
optimizer = optim.AdamW(model.parameters())
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, 1, total_steps=20)
trainer = Trainer(model, image_transforms, criterion, optimizer, scheduler, "Saved/SqueezeNet - Subset - Retrained/Model.tar", data)
train_test(trainer, "Saved/SqueezeNet - Subset - Retrained")

# ResNet50 model
print("\nTraining ResNet152 model - " + str(time.strftime("%Y-%m-%d %H:%M:%S")))
model = models.resnet152(pretrained=True)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 1000),
    nn.ReLU6(),
    nn.Dropout(0.2),
    nn.Linear(1000, 85)
)
optimizer = optim.AdamW(model.parameters())
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, 1, total_steps=20)
trainer = Trainer(model, image_transforms, criterion, optimizer, scheduler, "Saved/ResNet50 - Subset - Retrained/Model.tar", data)
train_test(trainer, "Saved/ResNet50 - Subset - Retrained")
