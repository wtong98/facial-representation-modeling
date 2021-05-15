"""
Exploratory analysis into why vae-512 looks so weird
"""

# <codecell>
import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split

import sys
sys.path.append('../')

from util import lda_test_acc, lda_analysis

mu_path = r'/home/grandpaa/workspace/yulab_face/pytorch_playground/save/vae/grid_search/vae512/mu_points.npy'
feats_path = r'/home/grandpaa/workspace/yulab_face/pytorch_playground/save/vae/grid_search/vae512/feats.pickle'
# <codecell>
mu_points = np.load(mu_path)
with open(feats_path, 'rb') as fp:
    feats = pickle.load(fp)

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
def analyze(class0, class1, title='Projection onto LDA Axis',
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
    plt.hist(mags_0, bins=_get_bins(mags_0.flatten()), alpha=0.6, label=name0)
    plt.hist(mags_1, bins=_get_bins(mags_1.flatten()), alpha=0.6, label=name1)
    # plt.axvline(x=center, color='red')
    plt.xlabel('Projection coordinate')
    plt.ylabel('Count')
    plt.legend()

    for i in range(1):
        train_dat, test_dat, train_labs, test_labs = train_test_split(data, labels, 
                                                                  test_size=0.2, 
                                                                  random_state=i+7)
        clf = LinearDiscriminantAnalysis()
        clf.fit(train_dat, train_labs)
        preds = clf.predict(test_dat)
        print(preds == test_labs)

        line = clf.coef_.reshape(-1, 1)
        mags_0 = test_dat[preds==0] @ line * (1 / np.linalg.norm(line))
        mags_1 = test_dat[preds==1] @ line * (1 / np.linalg.norm(line))

        plt.title(title)
        plt.hist(mags_0, bins=_get_bins(mags_0.flatten()), alpha=0.4, label=name0)
        plt.hist(mags_1, bins=_get_bins(mags_1.flatten()), alpha=0.4, label=name1)

def _get_bins(data):
    # h = 2 * iqr(data) / np.power(len(data), 1/3)
    # bins = int((np.max(data) - np.min(data)) / h)
    bins = int(len(data) / 12)
    return bins

analyze(male_points, female_points)

# <codecell>
def lda_test_acc(*classes, test_prop=0.2, seed=53110):
    data = np.concatenate(classes)
    class_labels = []
    for i, c in enumerate(classes):
        label = np.zeros(c.shape[0]) + i
        class_labels.append(label)
    
    labels = np.concatenate(class_labels)

    train_dat, test_dat, train_labs, test_labs = train_test_split(data, labels, 
                                                                  test_size=test_prop, 
                                                                  random_state=seed)
    clf = LinearDiscriminantAnalysis()
    clf.fit(train_dat, train_labs)
    preds = clf.predict(test_dat)
    print(preds == test_labs)

    line = clf.coef_.reshape(-1, 1)
    mags_0 = test_dat[preds==0] @ line * (1 / np.linalg.norm(line))
    mags_1 = test_dat[preds==1] @ line * (1 / np.linalg.norm(line))

    # plt.hist(mags_0, bins=_get_bins(mags_0.flatten()), alpha=0.8)
    # plt.hist(mags_1, bins=_get_bins(mags_1.flatten()), alpha=0.8)

    # NOTE: still a great deal of overlap, even if classes seem well separated
    # TODO: adjust bins to be more elegant
    mags_0 = test_dat[test_labs==0] @ line * (1 / np.linalg.norm(line))
    mags_1 = test_dat[test_labs==1] @ line * (1 / np.linalg.norm(line))

    plt.hist(mags_0, bins=20, alpha=0.8, label='Male, test')
    plt.hist(mags_1, bins=20, alpha=0.8, label='Female, test')

    mags_0 = train_dat[train_labs==0] @ line * (1 / np.linalg.norm(line))
    mags_1 = train_dat[train_labs==1] @ line * (1 / np.linalg.norm(line))

    plt.hist(mags_0, bins=10, alpha=0.8, label='Male, train')
    plt.hist(mags_1, bins=10, alpha=0.8, label='Female, train')

    plt.legend()
    plt.title('Training vs test data after LDA')
    plt.savefig('/tmp/haven/train_vs_test.png')

    return clf.score(test_dat, test_labs)

print(lda_test_acc(male_points, female_points))