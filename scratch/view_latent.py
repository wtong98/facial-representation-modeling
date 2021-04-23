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
from scipy.spatial import distance
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from tqdm import tqdm


# add neccessary modules
import sys
sys.path.append('../')

from dataset.cfd import build_datasets
# from dataset.utk import build_datasets
from model.vae import VAE
# from model.vae_gm import GMVAE

data_path = Path('../data/cfd')
# data_path = Path('../data/utk')
model_path = Path('../save/vae/vae_jan19_final.pt')
save_path = Path('../save/vae/cfd/latent')

# <codecell>
if not save_path.exists():
    save_path.mkdir(parents=True)

model = VAE()
ckpt = torch.load(model_path, map_location=torch.device('cpu'))
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

_, test_ds = build_datasets(data_path, train_test_split=1)

# test_ds = [test_ds[i] for i in range(10)]

# <codecell>
test_len = len(test_ds)
ldims = model.latent_dims
mu_points = np.zeros((test_len, ldims))
var_points = np.zeros((test_len, ldims))
feats = []

for i in tqdm(range(test_len)):
    im, feat = test_ds[i]
    im = im.unsqueeze(0)
    mu, var = model.encode(im)
    # mu, var = model.encode(im)[0]   # neccessary adaptation for GMVAE
    
    with torch.no_grad():
        mu_points[i] = mu.numpy()
        var_points[i] = var.numpy()

    feats.append(feat)

np.save(save_path / 'mu_points.npy', mu_points)
np.save(save_path / 'var_points.npy', var_points)
with open(save_path / 'feats.pickle', 'wb') as fp:
    pickle.dump(feats, fp)

# <codecell>
mu_points = np.load(save_path / 'mu_points.npy')
var_points = np.load(save_path / 'var_points.npy')
with open(save_path / 'feats.pickle', 'rb') as fp:
    feats = pickle.load(fp)

# <codecell>

#### BEGIN TSNE -------------------------------------------
# mu_points2d = TSNE().fit_transform(mu_points)
# np.save(save_path / 'mu_points2d.npy', mu_points2d)

# # <codecell>
# mu_points2d = np.load(save_path / 'mu_points2d.npy')

# # <codecell>
# # TODO: size to represent uncertainty?
# # all_colors = np.array([int(feat['ethnicity']) for feat in feats])
# all_colors = np.array([int(feat['ethnicity']) for feat in feats])
# idxs = [(val in (2, 3)) for val in all_colors]
# colors = all_colors[idxs]

# x = mu_points2d[idxs][:,0]
# y = mu_points2d[idxs][:,1]

# plt.title('VAE Latent Space with Ethnicity Labels (Black vs Asian)')
# scat = plt.scatter(x, y, c=colors, alpha=0.5)
# plt.legend(*scat.legend_elements(), title="Ethnicity", loc="upper right")
# # plt.show()
# plt.savefig(save_path / 'asian_black.png')

##### END TSNE #### ------------------------------------

# <codecell>
# Separate data points by label
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

##### END LDA ANALYSIS #### -----------------------------

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

# <codecell>
# TODO: redo VAE with permute instead of reshape
model = model.double()
with torch.no_grad():
    white_im = model.decode(torch.from_numpy(mean_white)).numpy()
    black_im = model.decode(torch.from_numpy(mean_black)).numpy()
    asian_im = model.decode(torch.from_numpy(mean_asian)).numpy()
    male_im = model.decode(torch.from_numpy(mean_male)).numpy()
    female_im = model.decode(torch.from_numpy(mean_female)).numpy()
    full_im = model.decode(torch.from_numpy(full_mean)).numpy()

white_im = np.squeeze(white_im).transpose(1, 2, 0)
plt.title('Average across White faces')
plt.imshow(white_im)
plt.savefig(save_path / 'white_mean.png')

black_im = np.squeeze(black_im).transpose(1, 2, 0)
plt.title('Average across Black faces')
plt.imshow(black_im)
plt.savefig(save_path / 'black_mean.png')

asian_im = np.squeeze(asian_im).transpose(1, 2, 0)
plt.title('Average across Asian faces')
plt.imshow(asian_im)
plt.savefig(save_path / 'asian_mean.png')

male_im = np.squeeze(male_im).transpose(1, 2, 0)
plt.title('Average across Male faces')
plt.imshow(male_im)
plt.savefig(save_path / 'male_mean.png')

female_im = np.squeeze(female_im).transpose(1, 2, 0)
plt.title('Average across Female faces')
plt.imshow(female_im)
plt.savefig(save_path / 'female_mean.png')

full_im = np.squeeze(full_im).transpose(1, 2, 0)
plt.title('Average across full face space')
plt.imshow(full_im)
plt.savefig(save_path / 'full_mean_face.png')

# TODO: do several asian reco's to confirm result

# <codecell>
line = (mean_white - mean_asian).reshape(-1, 1)
mags = mu_points @ line * (1 / np.linalg.norm(line))
center = np.mean(mags)

plt.title('Projection of faces onto White/Asian axis')
plt.hist(mags, bins=100)
plt.axvline(x=center, color='red')
plt.xlabel('<-- more Asian ------ more White -- >')
plt.savefig(save_path / 'mean_line_proj_hist_asian_white.png')

# <codecell>
line = (mean_white - mean_black).reshape(-1, 1)
mags = mu_points @ line * (1 / np.linalg.norm(line))
center = np.mean(mags)

plt.title('Projection of faces onto White/Black axis')
plt.hist(mags, bins=100)
plt.axvline(x=center, color='red')
plt.xlabel('<-- more Black ------ more White -- >')
plt.savefig(save_path / 'mean_line_proj_hist_black_white.png')

# <codecell>
line = (mean_asian - mean_black).reshape(-1, 1)
mags = mu_points @ line * (1 / np.linalg.norm(line))
center = np.mean(mags)

plt.title('Projection of faces onto Asian/Black axis')
plt.hist(mags, bins=100)
plt.axvline(x=center, color='red')
plt.xlabel('<-- more Black ------ more Asian -- >')
plt.savefig(save_path / 'mean_line_proj_hist_black_asian.png')

# <codecell>
line = (mean_female - mean_male).reshape(-1, 1)
mags = mu_points @ line * (1 / np.linalg.norm(line))
center = np.mean(mags)

plt.title('Projection of faces onto Male/Female axis')
plt.hist(mags, bins=100)
plt.axvline(x=center, color='red')
plt.xlabel('<-- more Male ------ more Female -- >')
plt.savefig(save_path / 'mean_line_proj_hist_male_female.png')
##### END MEAN POINT ### ---------------------------------

# <codecell>
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
plt.title('Separation between classes learned by VAE')
plt.savefig(save_path / 'd_prime_distance_between_classes.png')

###### END d' #### ------------------------------------

# <codecell>
# scratch
std_dev = np.sqrt(np.mean(full_var))
np.sqrt(np.linalg.norm(mean_white - mean_asian)) / std_dev
# %%
