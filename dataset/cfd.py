"""
Wrapper for CFD dataset. Labels are:

Ethnicity:
0: White
1: Black
2: Asian
3: Latinx

Sex: 
0: Male
1: Female

author: William Tong
date: 4/12/2021
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from skimage.transform import resize
from torch.utils.data import Dataset, random_split

from dataset.celeba import IM_DIMS

TOTAL_IMAGES = 597

class CFDDataset(Dataset):
    def __init__(self, im_path, total=TOTAL_IMAGES):
        self.im_path = im_path
        self.total = total
        self.files = list(im_path.iterdir())

    def __getitem__(self, idx):
        target_path = self.files[idx]
        im = plt.imread(target_path).astype('float32') / 255
        im = resize(im, IM_DIMS)
        im = im.transpose(2, 0, 1)

        labels = target_path.stem.split('-')

        features = {
            'gender': CFDDataset._gender_to_label(labels[1][1]),
            'ethnicity': CFDDataset._ethnicity_to_label(labels[1][0])
        }

        return torch.from_numpy(im), features
    
    @staticmethod
    def _gender_to_label(gender: str):
        if gender == 'M':
            return 0
        elif gender == 'F':
            return 1
        else:
            print('Unknown gender: ', gender)
            return -1
    
    @staticmethod
    def _ethnicity_to_label(eth: str):
        if eth == 'W':
            return 0
        elif eth == 'B':
            return 1
        elif eth == 'A':
            return 2
        elif eth == 'L':
            return 3
        else:
            print('Unknown ethnicity: ', eth)
            return -1 


    def __len__(self):
        return self.total


def build_datasets(im_path: Path, total=TOTAL_IMAGES, train_test_split=0.05, seed=53110) -> (Dataset, Dataset):
    if type(im_path) == str:
        im_path = Path(im_path)

    ds = CFDDataset(im_path, total)
    total = len(ds)

    num_test = int(total * train_test_split)
    num_train = total - num_test

    test_ds, train_ds = random_split(ds, (num_test, num_train), generator=torch.Generator().manual_seed(seed))
    return train_ds, test_ds