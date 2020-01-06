#!/usr/bin/env python
# coding: utf-8

from torch import nn, optim
from torchvision import transforms, models
from .Trainer import Trainer

class TransferLearning(Trainer):
    def __init__(self, model, criterion, optimizer, scheduler, path_saved="", feature_extractor = True, trainning=True):
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

        # Freeze all weights
        if feature_extractor == True:
            for param in model.parameters():
                param.requires_grad = False
        
        super().__init__(model, image_transforms, criterion, optimizer, scheduler, path_saved, trainning)
