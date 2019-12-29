#!/usr/bin/env python
# coding: utf-8

import numpy as np

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from SnakeDataset import SnakeDataset

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
