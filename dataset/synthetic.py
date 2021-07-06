
"""
Synthetic data generated from decoder of (V)AE

author: William Tong (wlt2115@columbia.edu)
date: 6/29/2021
"""

# TODO: compare to Hiden Manifold Model
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset, random_split

import sys
sys.path.append('../')

from model.ae import AE
from dataset.celeba import TOTAL_IMAGES as CELEBA_LEN

IM_DIMS = (218, 178)

class SyntheticDataset(Dataset):
    def __init__(self, model_path: Path, total_images=CELEBA_LEN, true_width=64, seed=None):
        self.model_path = model_path
        self.total_images = total_images
        self.decoder = None

        if seed == None:
            seed = total_images

        torch.manual_seed(seed)

        self.mu_0 = torch.ones(true_width)
        self.cov_0 = torch.eye(true_width)
        self.dist_0 = torch.distributions.multivariate_normal.MultivariateNormal(self.mu_0, self.cov_0)
        self.samp_0 = self.dist_0.sample((self.total_images,))

        self.mu_1 = -torch.ones(true_width)
        self.cov_1 = torch.eye(true_width)
        self.dist_1 = torch.distributions.multivariate_normal.MultivariateNormal(self.mu_1, self.cov_1)
        self.samp_1 = self.dist_1.sample((self.total_images,))

        self.decoder = Decoder(true_width)
        if not model_path.exists():
            self.decoder.reinit()
            torch.save(self.decoder.state_dict(), self.model_path)
        else:
            self.decoder.load_state_dict(torch.load(self.model_path)) # TODO: map to cpu / gpu?
        
        self.decoder.eval()
        

    def __getitem__(self, idx):
        with torch.no_grad():
            samp = self.samp_0 if idx % 2 == 0 else self.samp_1
            return torch.squeeze(self.decoder(samp[idx]))

    def __len__(self):
        return self.total_images


def build_datasets(model_path, total=CELEBA_LEN, train_test_split=0.01, seed=53110) -> Tuple[Dataset, Dataset]:
    ds = SyntheticDataset(model_path, total_images=total)
    total = len(ds)

    num_test = int(total * train_test_split)
    num_train = total - num_test

    test_ds, train_ds = random_split(ds, (num_test, num_train), generator=torch.Generator().manual_seed(seed))
    return train_ds, test_ds


class Decoder(nn.Module):
    def __init__(self, true_width=64):
        super(Decoder, self).__init__()

        center_size = 128*14*12
        self.fc_dec = nn.Linear(true_width, center_size)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=5,
                            stride=2, padding=2, output_padding=(1, 0)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=5,
                            stride=2, padding=2, output_padding=0),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=5,
                            stride=2, padding=2, output_padding=0),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=5,
                            stride=2, padding=2, output_padding=1),
            nn.Sigmoid()
        )
    

    def forward(self, z):
        x = self.fc_dec(z)
        x = x.view(-1, 128, 14, 12)
        x = self.decoder(x)
        return x
    

    def reinit(self):
        def re(m):
            if isinstance(m, nn.Conv2d):
                # TODO: compare to HMM
                nn.init.uniform_(m.weight, -1, 1)
                nn.init.uniform_(m.bias, -1, 1)

        self.decoder.apply(re)
        nn.init.uniform_(self.fc_dec.weight)
        nn.init.uniform_(self.fc_dec.bias)
        
