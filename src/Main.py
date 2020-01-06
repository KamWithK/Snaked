#!/usr/bin/env python
# coding: utf-8
import os, time, torch

from torchvision import models
from torch import nn, optim

from Models.TransferLearning import TransferLearning

def train_test(trainer, path):
    if not os.path.exists("Saved"):
        os.mkdir("Saved")
    if not os.path.exists(path):
        os.mkdir(path)
    trainer.train(path)

# Criterion
criterion = nn.CrossEntropyLoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MobileNet V2 model
print("Training MobileNet V2 model - " + str(time.strftime("%Y-%m-%d %H:%M:%S")))
model = models.mobilenet_v2(pretrained=True)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 85)
optimizer = optim.Adam(model.parameters())
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
trainer = TransferLearning(model, criterion, optimizer, scheduler, "Saved/MobileNetV2 - Subset - Retrained/Model.tar", False)
train_test(trainer, "Saved/MobileNetV2 - Subset - Retrained")

# SqueezeNet model
print("\nSqueezeNet 1_1 model - " + str(time.strftime("%Y-%m-%d %H:%M:%S")))
model = models.squeezenet1_1(pretrained=True)
model.classifier[1] = nn.Conv2d(512, 85, (1, 1), (1, 1))
optimizer = optim.Adam(model.parameters())
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
trainer = TransferLearning(model, criterion, optimizer, scheduler, "Saved/SqueezeNet - Subset - Retrained/Model.tar", False)
train_test(trainer, "Saved/SqueezeNet - Subset - Retrained")

# ResNet50 model
print("\nTraining ResNet50 model - " + str(time.strftime("%Y-%m-%d %H:%M:%S")))
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 85)
optimizer = optim.Adam(model.parameters())
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
trainer = TransferLearning(model, criterion, optimizer, scheduler, "Saved/ResNet50 - Subset - Retrained/Model.tar", False)
train_test(trainer, "Saved/ResNet50 - Subset - Retrained")
