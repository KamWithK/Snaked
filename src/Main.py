#!/usr/bin/env python
# coding: utf-8
import os, time, torch

from torchvision import models, transforms
from torch import nn, optim
from Models.Trainer import Trainer
from Data.Organiser import Organiser

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

def default_trainer(model, path, batch_size, lr=3e-3, find_lr=False):
    data_loaders = organiser.get_loaders(batch_size=batch_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters())
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, lr, epochs=100, steps_per_epoch=len(data_loaders["train"]))

    organiser.create_folder(path)

    trainer = Trainer(model, criterion, optimizer, scheduler, path, data_loaders)

    if find_lr == True:
        trainer.find_lr()

    return trainer

organiser = Organiser(data, image_transforms)

# MobileNet V2 model
print("Training MobileNet V2 model - " + str(time.strftime("%Y-%m-%d %H:%M:%S")))
model = models.mobilenet_v2(pretrained=True)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 85)
trainer = default_trainer(model, "Saved/MobileNetV2 - Retrained", 256, 5e-4)
trainer.train()

# SqueezeNet model
print("\nSqueezeNet 1_1 model - " + str(time.strftime("%Y-%m-%d %H:%M:%S")))
model = models.squeezenet1_1(pretrained=True)
model.classifier[1] = nn.Conv2d(512, 85, (1, 1), (1, 1))
trainer = default_trainer(model, "Saved/SqueezeNet - Retrained", 256)
trainer.train()

# ResNet152 model
print("\nTraining ResNet152 model - " + str(time.strftime("%Y-%m-%d %H:%M:%S")))
model = models.resnet152(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 85)
trainer = default_trainer(model, "Saved/ResNet152 - Retrained", 128, 1e-2)
trainer.train()
