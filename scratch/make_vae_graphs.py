"""
Make graphs comparing different VAEs under different hyperparameter
configs
"""
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial import distance
from tqdm import tqdm

import sys
sys.path.append('../')

from dataset.cfd import build_datasets
from model.vae import VAE
from util import lda_analysis, lda_test_acc, svd_analysis


out_path = Path('../save/vae/grid_search')
_, test_ds = build_datasets(Path('../data/cfd'), train_test_split=1)

class ModelData:
    def __init__(self, name: str, save_path: str, params: dict):
        self.name = name
        self.save_path = save_path
        self.params = params

# TODO: retrain VAE models with 2^n - 1 width
configs = [
    ModelData('vae64', '../save/vae/vae64.pt', { 'latent_dims': 64 }),
    ModelData('vae128', '../save/vae/vae128.pt', { 'latent_dims': 129 }),
    ModelData('vae256', '../save/vae/vae256.pt', { 'latent_dims': 257 }),
    ModelData('vae512', '../save/vae/vae512.pt', { 'latent_dims': 512 }),
    ModelData('vae1024', '../save/vae/vae1024.pt', { 'latent_dims': 1024 }),
    # ModelData('vae2048', '../save/vae/vae2048.pt', { 'latent_dims': 2048 }),
]


for conf in configs:
    print('processing conf: {}'.format(conf.name))
    out_dir = out_path / conf.name
    if not out_dir.exists():
        out_dir.mkdir(parents=True)

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

    if not (out_dir / 'mu_points.npy').exists():
        for i in tqdm(range(test_len)):
            im, feat = test_ds[i]
            im = im.unsqueeze(0)
            mu, var = model.encode(im)
            
            with torch.no_grad():
                mu_points[i] = mu.numpy()
                var_points[i] = var.numpy()

            feats.append(feat)

        np.save(out_dir / 'mu_points.npy', mu_points)
        np.save(out_dir / 'var_points.npy', var_points)
        with open(out_dir / 'feats.pickle', 'wb') as fp:
            pickle.dump(feats, fp)
    else:
        mu_points = np.load(out_dir / 'mu_points.npy')
        var_points = np.load(out_dir / 'var_points.npy')
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

    # lda analysis
    if not (out_dir / 'lda_analysis').exists():
        (out_dir / 'lda_analysis').mkdir()

    lda_analysis(male_points, female_points, title='Projection onto Male/Female LDA Axis',
                                            name0='Male',
                                            name1='Female',
                                            save_path=out_dir / 'lda_analysis' / 'male_female_lda.png')

    lda_analysis(black_points, white_points, title='Projection onto Black/White LDA Axis',
                                            name0='Black',
                                            name1='White',
                                            save_path=out_dir / 'lda_analysis' / 'black_white_lda.png')

    lda_analysis(black_points, asian_points, title='Projection onto Black/Asian LDA Axis',
                                            name0='Black',
                                            name1='Asian',
                                            save_path=out_dir / 'lda_analysis' / 'black_asian_lda.png')

    lda_analysis(asian_points, white_points, title='Projection onto Asian/White LDA Axis',
                                            name0='Asian',
                                            name1='White',
                                            save_path=out_dir / 'lda_analysis' / 'asian_white_lda.png')
    
    # LDA Test Error
    white_to_asian = lda_test_acc(white_points, asian_points)
    white_to_black = lda_test_acc(white_points,black_points)
    asian_to_black = lda_test_acc(asian_points, black_points)
    male_to_female = lda_test_acc(male_points, female_points)

    plt.bar(x = np.arange(4), height=[
        white_to_asian,
        white_to_black,
        asian_to_black,
        male_to_female
    ], tick_label=[
        'White to Asian',
        'White to Black',
        'Asian to Black',
        'Male to Female'
    ])
    plt.ylim(0.5, 1)
    plt.ylabel('Accuracy')
    plt.title('LDA Test Accuracy')
    plt.savefig(out_dir / 'lda_test_accuracy.png')
    plt.clf()

    # SVD Analysis
    svd_analysis(white_points, asian_points, black_points, 
                title='Singular Values of Ethnicity Classes',
                names=['White', 'Asian', 'Black'],
                save_path=out_dir / 'white_asian_black_sv.png')

    svd_analysis(male_points, female_points,  
                title='Singular Values of Gender Classes',
                names=['Male', 'Female'],
                save_path=out_dir / 'male_female_sv.png')
