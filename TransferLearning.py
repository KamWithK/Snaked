#!/usr/bin/env python
# coding: utf-8

from torch import nn, optim
from torchvision import transforms, models
from Trainer import Trainer

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

model = models.vgg16(pretrained=True)

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

trainer = Trainer(model, transforms, criterion, optimizer)

# Train
model, history = trainer.train("VGG16Model.pth")
