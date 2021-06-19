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

from dataset.cfd import build_datasets
from model.vae import VAE
from util import *


out_path = Path('../save/vae/dd_jun5')
_, test_ds = build_datasets(Path('../data/cfd'), train_test_split=1)

class ModelData:
    def __init__(self, name: str, save_path: str, params: dict):
        self.name = name
        self.save_path = save_path
        self.params = params

configs = [
    ModelData('vae64', '../save/vae/vae64.pt', { 'latent_dims': 64 }),
    ModelData('vae128', '../save/vae/vae128.pt', { 'latent_dims': 129 }),
    ModelData('vae256', '../save/vae/vae256.pt', { 'latent_dims': 257 }),
    ModelData('vae512', '../save/vae/vae512.pt', { 'latent_dims': 512 }),
    ModelData('vae1024', '../save/vae/vae1024.pt', { 'latent_dims': 1024 }),
    ModelData('vae2048', '../save/vae/vae2048.pt', { 'latent_dims': 2048 }),
    ModelData('vae4096', '../save/vae/vae4096.pt', { 'latent_dims': 4096 }),
    ModelData('vae8192', '../save/vae/vae8192.pt', { 'latent_dims': 8192 }),
    ModelData('vae16384', '../save/vae/vae16384.pt', { 'latent_dims': 16384 }),
]

white_to_asian_lda_acc = []
white_to_black_lda_acc = []
asian_to_black_lda_acc = []
male_to_female_lda_acc = []

white_to_asian_lda_acc_pca = []
white_to_black_lda_acc_pca = []
asian_to_black_lda_acc_pca = []
male_to_female_lda_acc_pca = []

for conf in configs:
    print('processing conf: {}'.format(conf.name))
    out_dir = out_path / conf.name
    if not out_dir.exists():
        out_dir.mkdir(parents=True)

    mu_points = None
    if not (out_dir / 'mu_points.npy').exists():
        # set up model
        model = VAE(**conf.params)
        ckpt = torch.load(conf.save_path, map_location=torch.device('cpu'))
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()

        # encode data
        test_len = len(test_ds)
        ldims = model.latent_dims
        mu_points = np.zeros((test_len, ldims))
        var_points = np.zeros((test_len, ldims))
        feats = []

        for i in tqdm(range(test_len)):
            im, feat = test_ds[i]
            im = im.unsqueeze(0)
            mu, var = model.encode(im)
            
            with torch.no_grad():
                mu_points[i] = mu.numpy()
                var_points[i] = var.numpy()

            feats.append(feat)

        np.save(out_dir / 'mu_points.npy', mu_points)
        with open(out_dir / 'feats.pickle', 'wb') as fp:
            pickle.dump(feats, fp)
    else:
        mu_points = np.load(out_dir / 'mu_points.npy')
        with open(out_dir / 'feats.pickle', 'rb') as fp:
            feats = pickle.load(fp)
    
    # stratify points
    all_colors = np.array([int(feat['ethnicity']) for feat in feats])
    all_genders = np.array([int(feat['gender']) for feat in feats])

    white_idxs = [val == 0 for val in all_colors]
    black_idxs = [val == 1 for val in all_colors]
    asian_idxs = [val == 2 for val in all_colors]

    male_idxs = [val == 0 for val in all_genders]
    female_idxs = [val == 1 for val in all_genders]

    white_points = mu_points[white_idxs]
    black_points = mu_points[black_idxs]
    asian_points = mu_points[asian_idxs]
    male_points = mu_points[male_idxs]
    female_points = mu_points[female_idxs]


    # BEGIN ANALYSIS RUNS ------------------------------------------------
    white_to_asian_lda_acc.append(lda_test_acc(white_points, asian_points))
    white_to_black_lda_acc.append(lda_test_acc(white_points,black_points))
    asian_to_black_lda_acc.append(lda_test_acc(asian_points, black_points))
    male_to_female_lda_acc.append(lda_test_acc(male_points, female_points))

    white_to_asian_lda_acc_pca.append(pca_double_descent_analysis(white_points, asian_points))
    white_to_black_lda_acc_pca.append(pca_double_descent_analysis(white_points,black_points))
    asian_to_black_lda_acc_pca.append(pca_double_descent_analysis(asian_points, black_points))
    male_to_female_lda_acc_pca.append(pca_double_descent_analysis(male_points, female_points))

    plot_test_hist(asian_points, white_points, title='Asian / White LDA Test Points',
                    name0='Asian', name1='White', save_path=out_dir / 'asian_white_lda_test_hist.png')

    plot_test_hist(black_points, white_points, title='Black / White LDA Test Points',
                    name0='Black', name1='White', save_path=out_dir / 'black_white_lda_test_hist.png')



# SUMMARY RUNS ------------------------------------------------------------
out_dir = out_path / 'fig'
if not out_dir.exists():
    out_dir.mkdir()

tick_labs = [model.name for model in configs]
# colors = ['blue', 'orange', 'green', 'red', 'purple']

# Original space
w2a_acc, w2a_err = zip(*white_to_asian_lda_acc)
w2b_acc, w2b_err = zip(*white_to_black_lda_acc)

plot_compare_bars(w2a_acc, w2a_err, w2b_acc, w2b_err, tick_labs,
                    fst_lab='White / Asian', sec_lab='Black / White',
                    ylab='Accuracy', title='LDA Test Accuracy', 
                    save_path=out_dir / 'asian_black_white_lda_acc.png')


w2a_acc, w2a_err, w2a_sv, w2a_sv_err = zip(*white_to_asian_lda_acc_pca)
w2b_acc, w2b_err, w2b_sv, w2b_sv_err = zip(*white_to_black_lda_acc_pca)

plot_compare_bars(w2a_acc, w2a_err, w2b_acc, w2b_err, tick_labs,
                    fst_lab='White / Asian', sec_lab='Black / White',
                    ylab='Accuracy', title='LDA Test Accuracy (Full PC Space)', 
                    save_path=out_dir / 'asian_black_white_lda_acc_pca.png')


# for i in range(len(w2a_sv)):
#     x = np.arange(len(w2a_sv[i]))
#     plt.errorbar(x, w2a_sv[i], fmt='-o', yerr=w2a_sv_err[i], color=colors[i])

# for i in range(len(w2b_sv)):
#     x = np.arange(len(w2b_sv[i]))
#     plt.errorbar(x, w2b_sv[i], fmt='--o', yerr=w2b_sv_err[i], color=colors[i])

# ax = plt.gca()
# for i in range(len(configs)):
#     rect = mpatches.Rectangle((0, 0), 0.001, 0.001, color=colors[i], label=tick_labs[i])
#     ax.add_patch(rect)


# plt.plot([0], [0], 'k-', label='Asian / White')
# plt.plot([0], [0], 'k--', label='Black / White')

# plt.title('Smallest SVs')
# plt.ylabel('Singular Value')
# plt.legend()
# plt.savefig(out_dir / 'asian_black_white_smallest_sv.png')
# plt.clf()
