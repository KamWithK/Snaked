#!/usr/bin/env python
# coding: utf-8

import os, torch

import numpy as np

from torchvision import transforms
from torch.utils.data import DataLoader, SubsetRandomSampler, WeightedRandomSampler
from Data.SnakeDataset import SnakeDataset

class Organiser():
    def __init__(self, data_map, transforms=transforms.ToTensor()):
        self.data_map = data_map
        self.transforms = transforms

    # Pass in a dictionary for data_map like the following (note that for full datasets positions DON'T have to be included):
    #data_map = {
        #"train": [path_to_data, path_to_csv, position_start, position_end],
        #"validation": [path_to_data, path_to_csv, position_start, position_end],
        #"test": [path_to_data, path_to_csv, position_start, position_end]
    #}
    def get_loaders(self, shuffle=True, batch_size=128, num_workers=5, auto_balance=False):
        self.data_loaders = {}
        self.data = {}

        for name in self.data_map:
            path_to_data = self.data_map[name][0]
            path_to_csv = self.data_map[name][1]
            self.data[name] = SnakeDataset(path_to_data, path_to_csv, self.transforms[name])
            self.data_loaders[name] = DataLoader(self.data[name], batch_size=batch_size, num_workers=num_workers),
            if len(self.data_map[name]) == 4:
                num_images, indices = len(self.data[name]), len(self.data[name])
                if shuffle == True:
                    # Randomly shuffle with set seed (for reproducability)
                    indices = np.random.RandomState(seed=11).permutation(num_images)
                position = [int(self.data_map[name][2] * num_images), int(self.data_map[name][3] * num_images)]

                # Training data must be over/under sampled
                # To ensure model learns to predict all classes
                item_weights = self.get_weights(indices[position[0]: position[1]])
                if name == "train" and auto_balance == True:
                    # Don't create weights for hole dataset, only training portion
                    # This prevents identical images being in different datasets
                    sampler = WeightedRandomSampler(torch.from_numpy(item_weights).double(), len(indices[position[0]: position[1]]))
                else:
                    sampler = SubsetRandomSampler(indices[position[0]: position[1]])

                self.data_loaders[name] = DataLoader(self.data[name], sampler=sampler, batch_size=batch_size, num_workers=num_workers)

        return self.data_loaders

    def get_weights(self, indices, phase="train"):
        associations = self.data[phase].targets
        self.label_counts = np.zeros(85)
        sample_weights = np.zeros(len(associations))

        # Find the number of samples of each class
        # Note that unique_values is an array of labels present AND their count
        unique_values = np.unique(associations[indices], return_counts=True)
        self.label_counts[unique_values[0]] = unique_values[1]

        # Labels with 0 samples are preset with a class weight of 0
        # Only set weights for samples where indices have been provided
        label_weights = np.divide(1.0, self.label_counts, out=np.zeros(len(self.label_counts)), where=self.label_counts!=0)
        sample_weights[indices] = label_weights[associations[indices]]

        # Note that weights PER SAMPLE are returned, NOT per class
        return sample_weights
    
    def create_folder(self, model_path: str):
        os.makedirs(model_path, exist_ok=True)
