"""
Common utilities used across scripts

author: William Tong
date: May 4, 2021
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import iqr

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def compute_lda_params(train_dat, train_labs):
    X0 = train_dat[train_labs==0]
    X1 = train_dat[train_labs==1]
    sig0 = X0.T @ X0
    sig1 = X1.T @ X1

    sig = (sig0 + sig1) / (X0.shape[0] + X1.shape[1])
    mu0 = np.mean(X0, axis=0).reshape(-1, 1)
    mu1 = np.mean(X1, axis=0).reshape(-1, 1)
    w = np.linalg.pinv(sig) @ (mu1 - mu0)  # NOTE: using pseudoinverse
    c = w.T @ ((1/2) * (mu0 + mu1))

    return w, c


def compute_lda_acc(train_dat, train_labs, test_dat, test_labs):
    w, c = compute_lda_params(train_dat, train_labs)

    Xp0 = test_dat[test_labs==0]
    Xp1 = test_dat[test_labs==1]
    score0 = Xp0 @ w + c
    score1 = Xp1 @ w + c

    acc0 = np.sum(score0 < 0) 
    acc1 = np.sum(score1 >= 0) 
    acc = (acc0 + acc1) / (Xp0.shape[0] + Xp1.shape[0])

    return acc


def compute_proj_coord(line, data):
    line = line.reshape(-1, 1)
    proj = data @ line   # NOTE: unnormalized
    return proj


def plot_test_hist(class0, class1, title='Test Histograms',
                    name0='Class 0', name1='Class 1', save_path=None,
                    test_prop=0.25, seed=53110):
    data = np.concatenate((class0, class1))
    labels = np.concatenate((np.zeros(class0.shape[0]), np.ones(class1.shape[0])))

    train_dat, test_dat, train_labs, test_labs = train_test_split(data, labels, 
                                        test_size=test_prop, random_state=seed)
    
    w, c = compute_lda_params(train_dat, train_labs)
    proj = compute_proj_coord(w, test_dat) + c

    mags_0 = proj[test_labs==0]
    mags_1 = proj[test_labs==1]

    plt.title(title)
    plt.hist(mags_0, bins=5, alpha=0.9, label=name0)
    plt.hist(mags_1, bins=5, alpha=0.9, label=name1)
    plt.xlabel('Score')
    plt.ylabel('Count')
    plt.legend()

    if save_path is not None:
        plt.savefig(save_path)
        plt.clf()
    else:
        plt.show()


# TODO: proj calculation is off by a factor
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

def lda_test_acc(*classes, test_prop=0.1, iter=5):
    data = np.concatenate(classes)
    class_labels = []
    for i, c in enumerate(classes):
        label = np.zeros(c.shape[0]) + i
        class_labels.append(label)
    
    labels = np.concatenate(class_labels)

    accs = []
    for _ in range(iter):
        train_dat, test_dat, train_labs, test_labs = train_test_split(data, labels, 
                                                                    test_size=test_prop)
        acc = compute_lda_acc(train_dat, train_labs, test_dat, test_labs)
        accs.append(acc)
    
    mean = np.mean(accs)
    std_err = np.std(accs) / np.sqrt(iter)
    
    return mean, std_err


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


def pca_double_descent_analysis(class0, class1, 
                                        num_sv=10,
                                        test_prop=0.1,
                                        iter=5,
                                        n_components=None):
    data = np.concatenate((class0, class1))
    labels = np.concatenate((np.zeros(class0.shape[0]), np.ones(class1.shape[0])))

    accs = []
    smallest_svs = []
    for _ in range(5):
        train_dat, test_dat, train_labs, test_labs = train_test_split(data, labels, 
                                                                    test_size=test_prop)

        pca = PCA(n_components=n_components, svd_solver='full')
        train_dat_pca = pca.fit_transform(train_dat)
        test_dat_pca = pca.transform(test_dat)

        acc = compute_lda_acc(train_dat_pca, train_labs, test_dat_pca, test_labs)
        accs.append(acc)
        smallest_sv = pca.singular_values_[-num_sv:]
        smallest_svs.append(smallest_sv)
    
    mean_acc = np.mean(accs)
    std_err_acc = np.std(accs) / np.sqrt(iter)

    smallest_svs = np.stack(smallest_svs, axis=0)
    mean_svs = np.mean(smallest_svs, axis=0)
    std_err_svs = np.std(smallest_svs, axis=0) / np.sqrt(iter)

    # TODO: bring new output to graphing
    return mean_acc, std_err_acc, mean_svs, std_err_svs


# TODO: change to new accordance
def pca_double_descent_analysis_logreg(class0, class1, 
                                       test_prop=0.2,
                                       seed=53110,
                                       n_components=None):
    data = np.concatenate((class0, class1))
    labels = np.concatenate((np.zeros(class0.shape[0]), np.ones(class1.shape[0])))

    pca = PCA(n_components=n_components)
    data_pca = pca.fit_transform(data)

    train_dat, test_dat, train_labs, test_labs = train_test_split(data_pca, labels, 
                                                                  test_size=test_prop, 
                                                                  random_state=seed)
    clf = LogisticRegression()
    clf.fit(train_dat, train_labs)
    acc = clf.score(test_dat, test_labs)
    # smallest_pc = pca.singular_values_[-1]
    
    sv = np.linalg.svd(test_dat, compute_uv=False)
    smallest_sv = sv[-1]

    return acc, smallest_sv


def bias_var_tradeoff_curve(class0, class1, steps=50,
                           title='Bias-Variance Tradeoff Curve',
                           save_path=None):
    data = np.concatenate((class0, class1))

    max_feats = np.min((data.shape[0] * 0.8, data.shape[1]))
    dims = np.floor(np.linspace(1, max_feats, steps)).astype('int')

    accs = []
    for dim in dims:
        acc, _ = pca_double_descent_analysis(class0, class1, n_components=dim)
        accs.append(1 - acc)
    
    plt.ylim(0, 0.7)
    plt.plot(dims, accs, '-o')
    plt.title(title)
    plt.xlabel('PCA Dimensions')
    plt.ylabel('LDA Test Error')
    
    if save_path is not None:
        plt.savefig(save_path)
        plt.clf()
    else:
        plt.plot()

def bias_var_tradeoff_curve_logreg(class0, class1, steps=50,
                           title='Bias-Variance Tradeoff Curve',
                           save_path=None):
    data = np.concatenate((class0, class1))

    max_feats = np.min(data.shape)
    dims = np.floor(np.linspace(1, max_feats, steps)).astype('int')

    accs = []
    for dim in dims:
        acc, _ = pca_double_descent_analysis_logreg(class0, class1, n_components=dim)
        accs.append(1 - acc)
    
    plt.ylim(0, 0.3)
    plt.plot(dims, accs, '-o')
    plt.title(title)
    plt.xlabel('PCA Dimensions')
    plt.ylabel('LogReg Test Error')
    
    if save_path is not None:
        plt.savefig(save_path)
        plt.clf()
    else:
        plt.plot()


def plot_compare_bars(fst_acc, fst_err, sec_acc, sec_err, tick_labs,
                        fst_lab='First', sec_lab='Second',
                        ylab='Count',
                        title='Comparative Barplot',
                        save_path=None):
    
    x = np.arange(len(fst_acc))
    width = 0.35

    fig, ax = plt.subplots()
    ax.bar(x - width/2, fst_acc, width, yerr=fst_err, label=fst_lab)
    ax.bar(x + width/2, sec_acc, width, yerr=sec_err, label=sec_lab)

    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labs)
    ax.legend()

    fig.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.clf()
    else:
        plt.show()
