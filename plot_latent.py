"""
Plots latent space for VAE.

author: William Tong
date 12/26/2020
"""

# <codecell>
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from dataset.utk import build_datasets
from model.vae import VAE

data_path = Path('data/utk')
model_path = Path('scratch/vae_save/epoch_47.pt')
save_path = Path('save/vae/latent')

if not save_path.exists():
    save_path.mkdir(parents=True)

vae = VAE().double()
ckpt = torch.load(model_path, map_location=torch.device('cpu'))
vae.load_state_dict(ckpt['model_state_dict'])
vae.eval()

_, test_ds = build_datasets(data_path, train_test_split=1)

test_ds = [test_ds[i] for i in range(10)]

# <codecell>
test_len = len(test_ds)
ldims = vae.latent_dims
mu_points = np.zeros((test_len, ldims))
var_points = np.zeros((test_len, ldims))
feats = []

for i in tqdm(range(test_len)):
    im, feat = test_ds[i]
    im = im.unsqueeze(0)
    mu, var = vae.encode(im)
    
    with torch.no_grad():
        mu_points[i] = mu.numpy()
        var_points[i] = var.numpy()

    feats.append(feat)

np.save(save_path / 'mu_points.npy', mu_points)
np.save(save_path / 'var_points.npy', var_points)
with open(save_path / 'feats.pickle', 'wb') as fp:
    pickle.dump(feats, fp)