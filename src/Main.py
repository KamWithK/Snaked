#!/usr/bin/env python
# coding: utf-8
import os

from torchvision import models
from torch import nn, optim

from Models.TransferLearning import TransferLearning

def create_folder_train(trainer, path):
    if not os.path.exists("Saved"):
        os.mkdir("Saved")
    if not os.path.exists(path):
        os.mkdir(path)
    trainer.train(path)

# Criterion
criterion = nn.CrossEntropyLoss()

# MobileNet V2 model
print("Training MobileNet V2 model")
model = models.mobilenet_v2(pretrained=True)
optimizer = optim.Adam(model.parameters())
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
trainer = TransferLearning(model, criterion, optimizer, scheduler, False)
trainer.model.classifier[1] = nn.Linear(trainer.model.classifier[1].in_features, 85)
create_folder_train(trainer, "Saved/MobileNetV2 - Subset - Retrained")

# SqueezeNet model
print("\nSqueezeNet 1_1 model")
model = models.squeezenet1_1(pretrained=True)
optimizer = optim.Adam(model.parameters())
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
trainer = TransferLearning(model, criterion, optimizer, scheduler, False)
trainer.model.classifier[1] = nn.Conv2d(512, 85, (1, 1), (1, 1))
create_folder_train(trainer, "Saved/SqueezeNet - Subset - Retrained")

# ResNet50 model
print("\nTraining ResNet50 model")
model = models.resnet50(pretrained=True)
optimizer = optim.Adam(model.parameters())
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
trainer = TransferLearning(model, criterion, optimizer, scheduler, False)
trainer.model.fc = nn.Linear(trainer.model.fc.in_features, 85)
create_folder_train(trainer, "Saved/ResNet50 - Subset - Retrained")
