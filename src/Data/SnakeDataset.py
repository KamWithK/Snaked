#!/usr/bin/env python
# coding: utf-8

import torch

import pandas as pd

from typing import NamedTuple
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

class Item(NamedTuple):
    index: int
    img: torch.Tensor
    species_number: int
    species: str
    file_name: str

class SnakeDataset(Dataset):
    # Create/Initialize variables
    def __init__(self, img_dir, csv_path, transforms=None):
        self.df = pd.read_csv(csv_path, usecols=["scientific_name", "filename"])
        self.df["species_num"] = self.df["scientific_name"].astype("category").cat.codes
        self.csv_path = csv_path
        self.img_dir = img_dir
        self.transforms = transforms
        self.targets = self.df["species_num"].to_numpy()
    
    def __len__(self):
        return len(self.df)
    
    # Return all information about a data point
    # Use a Tuple to remain Pythonic with feature growth
    def __getitem__(self, index):
        img = Image.open(self.img_dir + "/" + self.df["filename"][index]).convert("RGB")
        img = self.transforms(img)

        number = self.df["species_num"][index]

        return Item(index, img, number.astype("long"), self.df["scientific_name"][index], self.df["filename"][index])
