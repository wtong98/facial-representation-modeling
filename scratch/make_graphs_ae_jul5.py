"""
Making graphs per Jun 5 email and instructions
"""

import pickle
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
import matplotlib.patches as mpatches

import sys
sys.path.append('../')

from dataset.cfd import TOTAL_IMAGES
from dataset.synthetic import build_datasets
from model.ae import AE
from util import *


out_path = Path('../save/ae/dd_syn_jul5')
_, test_ds = build_datasets(Path('../save/syn_model'), total=TOTAL_IMAGES, return_labels=True, train_test_split=1)


class ModelData:
    def __init__(self, name: str, save_path: str, params: dict):
        self.name = name
        self.save_path = save_path
        self.params = params


configs = [
    ModelData('vae64', '../save/ae64_syn_jul3/final.pt', {'latent_dims': 64}),
    # ModelData('vae128', '../save/ae128_syn_jul3/final.pt', {'latent_dims': 127}),
    # ModelData('vae256', '../save/ae256_syn_jul3/final.pt', {'latent_dims': 255}),
    # ModelData('vae512', '../save/ae512_syn_jul3/final.pt', {'latent_dims': 511}),
    # ModelData('vae1024', '../save/ae1024_syn_jul3/final.pt',
    #           {'latent_dims': 1023}),
    # ModelData('vae2048', '../save/ae2048_syn_jul3/final.pt',
    #           {'latent_dims': 2048}),
    # ModelData('vae4096', '../save/ae4096_syn_jul3/final.pt',
    #           {'latent_dims': 4096}),
    # ModelData('vae8192', '../save/ae8192_syn_jul3/final.pt',
    #           {'latent_dims': 8192}),
    # ModelData('vae16384', '../save/ae16384_syn_jul3/final.pt',
    #           {'latent_dims': 16384}),
]

# For convenience, assuming 0 = male and 1 = female
male_to_female_lda_acc = []
male_to_female_lda_acc_pca = []

for conf in configs:
    print('processing conf: {}'.format(conf.name))
    out_dir = out_path / conf.name
    if not out_dir.exists():
        out_dir.mkdir(parents=True)

    mu_points = None
    if not (out_dir / 'mu_points.npy').exists():
        # set up model
        model = AE(**conf.params)
        ckpt = torch.load(conf.save_path, map_location=torch.device('cpu'))
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()

        # encode data
        test_len = len(test_ds)
        ldims = model.latent_dims
        mu_points = np.zeros((test_len, ldims))
        feats = []

        for i in tqdm(range(test_len)):
            im, feat = test_ds[i]
            im = im.unsqueeze(0)
            mu = model.encode(im)

            with torch.no_grad():
                mu_points[i] = mu.numpy()

            feats.append(feat)

        np.save(out_dir / 'mu_points.npy', mu_points)
        with open(out_dir / 'feats.pickle', 'wb') as fp:
            pickle.dump(feats, fp)
    else:
        mu_points = np.load(out_dir / 'mu_points.npy')
        with open(out_dir / 'feats.pickle', 'rb') as fp:
            feats = pickle.load(fp)

    # stratify points
    male_idxs = [val == 0 for val in feats]
    female_idxs = [val == 1 for val in feats]

    male_points = mu_points[male_idxs]
    female_points = mu_points[female_idxs]

    # BEGIN ANALYSIS RUNS ------------------------------------------------
    male_to_female_lda_acc.append(lda_test_acc(male_points, female_points))

    male_to_female_lda_acc_pca.append(
        pca_double_descent_analysis(male_points, female_points))

    plot_test_hist(male_points, female_points, title='LDA Test Points',
                   name0='Class 0', name1='Class 1', save_path=out_dir / 'lda_test_hist.png')

# SUMMARY RUNS ------------------------------------------------------------
out_dir = out_path / 'fig'
if not out_dir.exists():
    out_dir.mkdir()

tick_labs = [model.name for model in configs]
colors = ['blue', 'orange', 'green', 'red', 'purple', 'yellow']

plt.figure(figsize=(8, 6), dpi=150)

# Original space
acc, err = zip(*male_to_female_lda_acc)
plot_err_bars(acc, err, tick_labs, title='AE Error on Synthetic Data', save_path=out_dir / 'error_lda.png')

acc, err, sv, sv_err = zip(*male_to_female_lda_acc_pca)
plot_err_bars(acc, err, tick_labs, title='AE Error on Synthetic Data (Full PC Space)', save_path=out_dir / 'error_lda_pca.png')

sv = sv[-len(colors):]
sv_err = sv_err[-len(colors):]

for i in range(len(sv)):
    x = np.arange(len(sv[i]))
    plt.errorbar(x, sv[i], fmt='-o', yerr=sv_err[i], color=colors[i])

for i in range(len(sv)):
    x = np.arange(len(sv[i]))
    plt.errorbar(x, sv[i], fmt='--o', yerr=sv_err[i], color=colors[i])

ax = plt.gca()
for i in range(len(configs)):
    rect = mpatches.Rectangle((0, 0), 0.001, 0.001, color=colors[i], label=tick_labs[i])
    ax.add_patch(rect)


plt.plot([0], [0], 'k-', label='Asian / White')
plt.plot([0], [0], 'k--', label='Black / White')

plt.title('Smallest SVs')
plt.ylabel('Singular Value')
plt.legend()
plt.savefig(out_dir / 'smallest_sv.png')
plt.clf()
