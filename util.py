"""
Common utilities used across scripts

author: William Tong
date: May 4, 2021
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import iqr

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split


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
    return clf.score(test_dat, test_labs)


# TODO: standardize features, then SVD?
# TODO: how to generalize threshold to multiclass classification problem?
def svd_analysis(*classes, zoom=True,
                           title='Singular Values Plot',
                           names=None,
                           save_path=None):
    assert len(classes[0].shape) == 2, 'points must be a 2D array'

    svs = [np.linalg.svd(c, compute_uv=False) for c in classes]
    xlabs = np.arange(classes[0].shape[1])
    
    for i, sv in enumerate(svs):
        lab = names[i] if names is not None else 'Class {}'.format(i)
        plt.plot(xlabs[:len(sv)], sv, label=lab)
    
    plt.title(title)
    plt.xlabel('Singular value')
    plt.legend()

    if save_path is not None:
        plt.savefig(save_path)
        plt.clf()
    else:
        plt.plot()
    
    if zoom:
        for i, sv in enumerate(svs):
            # TODO: more gentle, dynamic way of zooming?
            lab = names[i] if names is not None else 'Class {}'.format(i)
            plt.plot(xlabs[len(sv)-20:len(sv)], sv[-20:], label=lab)
        
        plt.title(title + ' (zoomed)')
        plt.xlabel('Singular value')
        plt.legend()

        if save_path is not None:
            plt.savefig(str(save_path) + '.zoom.png')
            plt.clf()
        else:
            plt.plot()


    