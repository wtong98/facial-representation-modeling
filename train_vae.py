"""
Simple script for training a VAE on CelebA. Formatted as Hydrogen notebook.

author: William Tong (wlt2115@columbia.edu)
date: 10/20/2020
"""

#<codecell>
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

import torch
from torch import nn
from torch.nn import functional as F
from torch import optim

IM_DIMS = (28, 28)
SAVE_PATH = Path('save/vae_model/')

if not SAVE_PATH.exists():
    SAVE_PATH.mkdir(parents=True)