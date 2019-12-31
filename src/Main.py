#!/usr/bin/env python
# coding: utf-8

from torchvision import models
from torch import nn, optim

from Models.TransferLearning import TransferLearning

# VGG16 model training
vgg16_model = models.vgg16(pretrained=True)
criterion = nn.NLLLoss()
optimizer = optim.Adam(vgg16_model.parameters())
vgg_trainer = TransferLearning(vgg16_model, criterion, optimizer)
vgg_trainer.train("Saved/VGG16 - 1.pth")
