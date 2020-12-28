"""
Wrapper for UTK dataset

Ethnicity labels appear to be
0: White
1: Black
2: Asian
3: Indian
4: Other

Sex labels appear to be
0: Male
1: Female

author: William Tong
date: 12/26/2020
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from skimage.transform import resize
from torch.utils.data import DataLoader, Dataset, random_split

from dataset.celeba import IM_DIMS

# TODO: resize image, afix labels, test to ensure works
TOTAL_IMAGES = 23708

class UTKDataset(Dataset):
    def __init__(self, im_path, total=TOTAL_IMAGES):
        self.im_path = im_path
        self.total = total
        self.files = list(im_path.iterdir())

    def __getitem__(self, idx):
        target_path = self.files[idx]
        im = plt.imread(target_path)
        im = resize(im, IM_DIMS)
        im = im.reshape(-1, *IM_DIMS)

        labels = target_path.stem.split('_')

        features = {
            'age': labels[0],
            'gender': labels[1],
            'ethnicity': labels[2] if len(labels) == 4 else 4
        }

        return torch.from_numpy(im), features


    def __len__(self):
        return self.total


def build_datasets(im_path: Path, total=TOTAL_IMAGES, train_test_split=0.05, seed=53110) -> (Dataset, Dataset):
    if type(im_path) == str:
        im_path = Path(im_path)

    ds = UTKDataset(im_path, total)
    total = len(ds)

    num_test = int(total * train_test_split)
    num_train = total - num_test

    test_ds, train_ds = random_split(ds, (num_test, num_train), generator=torch.Generator().manual_seed(seed))
    return train_ds, test_ds