"""
Simple wrapper for mnist dataset. Unlike the CelebA dataset, it will load
images directly into memory. Labels are discarded. Training and testing images
are combined into one dataset, which the dataset builder will then resplit into
train-test categories based on the indicated ratio.

author: William Tong
date: 12/1/2020
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.io import loadmat
from torch.utils.data import DataLoader, Dataset, random_split


class MNIST(Dataset):
    def __init__(self, mnist_path, color=False):
        mnist = loadmat(mnist_path)
        first = np.float32(mnist['trainX']) / 255
        second = np.float32(mnist['testX']) / 255
        self.data = np.concatenate((first, second))

        if color:
            self.data = self.data.reshape(-1, 28, 28)
            self.data = np.stack((self.data, self.data, self.data), axis=1)


    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx])

    def __len__(self):
        return self.data.shape[0]


def build_datasets(im_path: Path, color=False, train_test_split=0.01, seed=53110):
    if type(im_path) == str:
        im_path = Path(im_path)

    ds = MNIST(im_path, color)
    total = len(ds)

    num_test = int(total * train_test_split)
    num_train = total - num_test

    test_ds, train_ds = random_split(ds, (num_test, num_train), generator=torch.Generator().manual_seed(seed))
    return train_ds, test_ds