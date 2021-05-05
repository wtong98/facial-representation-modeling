"""
Make the visualizations of the AAM space

author: William Tong
date: 4/14/2021
"""

# <codecell>
import math
from pathlib import Path

import mat73
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import sys
sys.path.append('../')

from dataset.cfd import CFDDataset

# <codecell>
# TODO: confirm that matlab reads files sequentially --> if not, enforce this
aam = mat73.loadmat('../data/aam/CFDnr_OFD_RGB_new_PCA_AF.mat')
with open('../data/cfd_names.txt') as fp:
    file_names = fp.readlines()

save_path = Path('../save/aam')

# <codecell>
shape_pcs = aam['shape']['PC']
texture_pcs = aam['texture']['PC']

shape_feat = shape_pcs[:,:30]
tex_red_feat = texture_pcs[:,:20,0]
tex_grn_feat = texture_pcs[:,:20,1]
tex_blu_feat = texture_pcs[:,:20,2]

shape_coord = aam['shape']['value'] @ shape_feat
tex_red_coord = aam['texture']['value'][:,:,0] @ tex_red_feat
tex_grn_coord = aam['texture']['value'][:,:,1] @ tex_grn_feat
tex_blu_coord = aam['texture']['value'][:,:,2] @ tex_blu_feat

face_coord = np.concatenate((shape_coord, tex_red_coord, tex_grn_coord, tex_blu_coord), axis=1)

def _make_feat(name: str) -> dict:
    labels = name.split('-')
    return {
        'gender': CFDDataset._gender_to_label(labels[1][1]),
        'ethnicity': CFDDataset._ethnicity_to_label(labels[1][0])
    }

feats = [_make_feat(name) for name in file_names]

# <codecell>
# --- Bottom shamelessly copied from view_latent.py
mu_points = face_coord   # small adaptation to make things work

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

# <codecell>
##### BEGIN LDA ANALYSIS ### ----------------------------
def lda_analysis(class0, class1, title='Projection onto LDA Axis',
                                 name0='Class 0',
                                 name1='Class 1',
                                 save_path=None):
    data = np.concatenate((class0, class1))
    labels = np.concatenate((np.zeros(class0.shape[0]), np.ones(class1.shape[0])))

    clf = LinearDiscriminantAnalysis()
    clf.fit(data, labels)

    line = clf.coef_.reshape(-1, 1)
    mags_0 = class0 @ line * (1 / np.linalg.norm(line))
    mags_1 = class1 @ line * (1 / np.linalg.norm(line))
    data = data @ line * (1 / np.linalg.norm(line))
    # center = np.mean(data)

    # TODO: correct bins
    plt.title(title)
    plt.hist(data, bins=100, alpha=0.2, label='All points')
    plt.hist(mags_0, bins=100, alpha=0.9, label=name0)
    plt.hist(mags_1, bins=100, alpha=0.9, label=name1)
    # plt.axvline(x=center, color='red')
    plt.xlabel('Projection coordinate')
    plt.ylabel('Count')
    plt.legend()

    if save_path is not None:
        plt.savefig(save_path)


lda_analysis(male_points, female_points, title='Projection onto Male/Female LDA Axis',
                                         name0='Male',
                                         name1='Female',
                                         save_path=save_path / 'lda_analysis' / 'male_female_lda.png')
plt.clf()

lda_analysis(black_points, white_points, title='Projection onto Black/White LDA Axis',
                                         name0='Black',
                                         name1='White',
                                         save_path=save_path / 'lda_analysis' / 'black_white_lda.png')
plt.clf()

lda_analysis(black_points, asian_points, title='Projection onto Black/Asian LDA Axis',
                                         name0='Black',
                                         name1='Asian',
                                         save_path=save_path / 'lda_analysis' / 'black_asian_lda.png')
plt.clf()


lda_analysis(asian_points, white_points, title='Projection onto Asian/White LDA Axis',
                                         name0='Asian',
                                         name1='White',
                                         save_path=save_path / 'lda_analysis' / 'asian_white_lda.png')
plt.clf()

# <codecell>
##### BEGIN MEAN POINT ### ------------------------------
# MEAN AND VAR CALCS (^_^)

mean_white = np.mean(white_points, axis=0)
mean_black = np.mean(black_points, axis=0)
mean_asian = np.mean(asian_points, axis=0)
mean_male = np.mean(male_points, axis=0)
mean_female = np.mean(female_points, axis=0)

var_white = np.cov(white_points, rowvar=False)
var_black = np.cov(black_points, rowvar=False)
var_asian = np.cov(asian_points, rowvar=False)
var_male = np.cov(male_points, rowvar=False)
var_female = np.cov(female_points, rowvar=False)

full_mean = np.mean(mu_points, axis=0)
full_var = np.cov(mu_points, rowvar=False)

###### BEGIN d' ### -----------------------------------
white_to_asian = distance.mahalanobis(mean_white, mean_asian, 0.5 * (var_white + var_asian))
print(white_to_asian)

white_to_black = distance.mahalanobis(mean_white, mean_black, 0.5 * (var_white + var_black))
print(white_to_black)

asian_to_black = distance.mahalanobis(mean_asian, mean_black, 0.5 * (var_asian + var_black))
print(asian_to_black)

male_to_female = distance.mahalanobis(mean_male, mean_female, 0.5 * (var_male + var_female))
print(male_to_female)

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
plt.ylabel('$d\'$')
plt.title('Separation between classes learned by AAM')
plt.savefig(save_path / 'd_prime_distance_between_classes.png')

# %%
