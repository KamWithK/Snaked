#!/usr/bin/env python
# coding: utf-8

import os

import numpy as np

from torchvision import transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
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
    def get_loaders(self, shuffle=True, batch_size=128, num_workers=5):
        data_loaders = {}

        for name in self.data_map:
            path_to_data = self.data_map[name][0]
            path_to_csv = self.data_map[name][1]
            data = SnakeDataset(path_to_data, path_to_csv, self.transforms[name])
            data_loaders[name] = DataLoader(data, batch_size=batch_size, num_workers=num_workers),
            if len(self.data_map[name]) == 4:
                num_images, indicies = len(data), len(data)
                if shuffle == True:
                    # Randomly shuffle with set seed (for reproducability)
                    indicies = np.random.RandomState(seed=11).permutation(num_images)
                position = [int(self.data_map[name][2] * num_images), int(self.data_map[name][3] * num_images)]
                sampler = SubsetRandomSampler(indicies[position[0]: position[1]])
                data_loaders[name] = DataLoader(data, sampler=sampler, batch_size=batch_size, num_workers=num_workers)
        return data_loaders
    
    def create_folder(self, model_path: str):
        os.makedirs(model_path, exist_ok=True)
