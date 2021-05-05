"""
Common utilities used across scripts

author: William Tong
date: May 4, 2021
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import iqr

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


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
    plt.hist(data, bins=_get_bins(data.flatten()), alpha=0.2, label='All points')
    plt.hist(mags_0, bins=_get_bins(mags_0.flatten()), alpha=0.9, label=name0)
    plt.hist(mags_1, bins=_get_bins(mags_1.flatten()), alpha=0.9, label=name1)
    # plt.axvline(x=center, color='red')
    plt.xlabel('Projection coordinate')
    plt.ylabel('Count')
    plt.legend()

    if save_path is not None:
        plt.savefig(save_path)
        plt.clf()


def _get_bins(data):
    # h = 2 * iqr(data) / np.power(len(data), 1/3)
    # bins = int((np.max(data) - np.min(data)) / h)
    bins = int(len(data) / 12)
    return bins
    