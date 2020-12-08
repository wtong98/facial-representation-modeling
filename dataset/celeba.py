"""
Defines a simple PyTorch dataset object for the CelebA face dataset.

author: William Tong (wlt2115@columbia.edu)
date: 11/5/2020
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset, random_split

IM_DIMS = (218, 178)
TOTAL_IMAGES = 202599

class CelebADataset(Dataset):
    def __init__(self, im_path, total=TOTAL_IMAGES):
        self.im_path = im_path
        self.total = total

    def __getitem__(self, idx):
        name = str(idx + 1).zfill(6) + '.jpg'
        target_path = self.im_path / name

        im = plt.imread(target_path).reshape(-1, *IM_DIMS)
        im = im.astype('float32') / 255
        return torch.from_numpy(im)

    def __len__(self):
        return self.total


def build_datasets(im_path: Path, total=TOTAL_IMAGES, train_test_split=0.01, seed=53110) -> (Dataset, Dataset):
    if type(im_path) == str:
        im_path = Path(im_path)

    ds = CelebADataset(im_path, total)
    total = len(ds)

    num_test = int(total * train_test_split)
    num_train = total - num_test

    test_ds, train_ds = random_split(ds, (num_test, num_train), generator=torch.Generator().manual_seed(seed))
    return train_ds, test_ds