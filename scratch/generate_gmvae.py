"""
Quick-n-dirty script for generating some samples from the GMVAE model

author: William Tong (wlt2115@columbia.edu)
date: 3/4/2021
"""

# <codecell>
from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np

import torch

# add neccessary modules
import sys
sys.path.append('../')

from model.vae_gm import GMVAE
from dataset.cfd import build_datasets
# from dataset.celeba import build_datasets
# from dataset.celeba_single import build_datasets
# from dataset.mnist import build_datasets

IM_DIMS = (218, 178)
TOTAL_IMAGES = 202599
MODEL_PATH = Path('../save/gmvae/final.pt')
DATA_PATH = Path('../data')
# IM_PATH = DATA_PATH / 'img'
IM_PATH = DATA_PATH / 'cfd'

# <codecell>
train_ds, test_ds = build_datasets(IM_PATH)

# <codecell>
model = GMVAE()
ckpt = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
model.load_state_dict(ckpt['model_state_dict'])

model.eval()

# <codecell>
idx = 5
samp_im = test_ds[idx].numpy().transpose(1, 2, 0)

plt.imshow(samp_im)
plt.show()

# <codecell>
idx = 5
samp = test_ds[idx].unsqueeze(0)
print(samp.shape)

with torch.no_grad():
    reco = model.reconstruct(samp)
    print(reco.shape)

    reco_im = torch.squeeze(reco).numpy().transpose(1,2,0)
    samp_im = torch.squeeze(samp).numpy().transpose(1,2,0)
    print(reco_im.shape)
    print(samp_im.shape)

plt.imshow(samp_im)
plt.show()
plt.imshow(reco_im)
plt.show()

# <codecell>
samp = [test_ds[i] for i in range(5)]   # index slices won't work on ds
samp = np.stack(samp)
samp = torch.from_numpy(samp)

with torch.no_grad():
    reco = model.reconstruct(samp)
    reco_im = torch.squeeze(reco).numpy().transpose(0, 2, 3, 1)
    samp_im = torch.squeeze(samp).numpy().transpose(0, 2, 3, 1)

combined = np.empty((reco_im.shape[0] + samp_im.shape[0], 218, 178, 3))
# combined = np.empty((reco_im.shape[0] + samp_im.shape[0], 28, 28, 3))
combined[0::2] = samp_im
combined[1::2] = reco_im

fig = plt.figure(figsize=(10, 10))
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(5, 2),
                 axes_pad=0.1,
                 )

for ax, im in zip(grid, combined):
    ax.imshow(im)

fig.suptitle('GMVAE reconstructions')
# plt.show()
plt.savefig('image/gmvae/gmvae_face_reco.png')

# <codecell>
for cluster_id in range(4):
    with torch.no_grad():
        samp = model.sample(25, cluster_id)
        samp_im = torch.squeeze(samp).numpy().transpose(0, 2, 3, 1)

    fig = plt.figure(figsize=(10, 10))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                    nrows_ncols=(5, 5),  # creates 2x2 grid of axes
                    axes_pad=0.1,  # pad between axes in inch.
                    )

    for ax, im in zip(grid, samp_im):
        ax.imshow(im)

    fig.suptitle('Sample faces drawn from GMVAE: Cluster %d' % cluster_id)
    plt.savefig('image/gmvae/gmvae_face_sample_id%d.png' % cluster_id)
# plt.show()
