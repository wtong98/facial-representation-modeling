"""
Quick-n-dirty script for generating some samples from VAE model

author: William Tong (wlt2115@columbia.edu)
date: 10/26/2019
"""

# <codecell>
from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
from scipy.io import loadmat

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import ParameterList, functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, Dataset, random_split

IM_DIMS = (218, 178)
TOTAL_IMAGES = 202599
MODEL_PATH = Path('../save/hm_bin/epoch_36.pt')
DATA_PATH = Path('../data/')
# IM_PATH = DATA_PATH / 'img'
IM_PATH = Path('../data/img')

train_test_split = 0.01

# <codecell>
def HM(color=True, layers=None):
    if color:
        return HM_color(layers)
    else:
        return HM_bw(layers)


class HM_bw(nn.Module):

    def __init__(self, layers=None):
        super(HM_bw, self).__init__()

        if layers is None:
            # self.layers = [784, 256, 64, 32]
            self.layers = [38804, 2048, 128, 32]
        else:
            self.layers = layers

        self.num_layers = len(self.layers) - 1

        # recognition layers
        for ii, i in enumerate(range(self.num_layers)):
            self.set_r(ii, nn.Linear(self.layers[i], self.layers[i+1]))

        # generation layers
        for ii, i in enumerate(range(self.num_layers)[::-1]):
            self.set_g(ii, nn.Linear(self.layers[i+1], self.layers[i]))

        self.g_bias = Parameter(torch.FloatTensor(self.layers[-1]))

        self.reset_parameters()

    def reset_parameters(self):
        self.g_bias.data.uniform_(-1, 1)

    def r(self, i):
        return getattr(self, "recognition_{}".format(i))

    def set_r(self, i, layer):
        setattr(self, "recognition_{}".format(i), layer)

    def g(self, i):
        return getattr(self, "generation_{}".format(i))

    def set_g(self, i, layer):
        setattr(self, "generation_{}".format(i), layer)

    def layer_output(self, x, training=True):
        """
        If training, treat x as bernoulli distribution and sample output,
        otherwise simply round x, giving binary output in either case.
        """
        # if training:
        #     out = torch.bernoulli(x).detach()
        # else:
        #     out = torch.round(x)
        out = torch.bernoulli(x).detach()
        return out

    def _run_wake_recognition(self, x):
        results = []

        # Run recognition layers, saving stochastic outputs.
        for i in range(self.num_layers):
            x = self.r(i)(x)
            x = torch.sigmoid(x)
            x = self.layer_output(x, self.training)
            results.append(x)

        return results

    def _run_wake_generation(self, x_original, recognition_outputs):
        results = []

        # Run generative layers, predicting the input to each layer.
        for i in range(self.num_layers):
            x_input = recognition_outputs[-(i+1)]
            if i == self.num_layers - 1:
                x_target = x_original
            else:
                x_target = recognition_outputs[-(i+2)]
            x = self.g(i)(x_input)
            x = torch.sigmoid(x)
            results.append(nn.BCELoss()(x, x_target))

        return results

    def run_wake(self, x):
        x_first = x
        batch_size = x.size(0)

        # Run Recognition Net.
        recognition_outputs = self._run_wake_recognition(x)

        # Fit the bias to the final layer.
        x_last = recognition_outputs[-1]
        x = self.g_bias.view(1, -1).expand(batch_size, self.g_bias.size(0))
        x = torch.sigmoid(x)
        generation_bias_loss = nn.BCELoss()(x, x_last)

        # Run Generation Net.
        generation_loss = self._run_wake_generation(x_first, recognition_outputs)

        return recognition_outputs, generation_bias_loss, generation_loss

    def _run_sleep_recognition(self, x_initial, generative_outputs):
        results = []

        # Run recognition layers to predict fantasies.
        for i in range(self.num_layers):
            x_input = generative_outputs[-(i+1)]
            if i == self.num_layers - 1:
                x_target = x_initial
            else:
                x_target = generative_outputs[-(i+2)]
            x = self.r(i)(x_input)
            x = torch.sigmoid(x)
            results.append(nn.BCELoss()(x, x_target))

        return results

    def _run_sleep_generation(self, x_initial):
        results = []

        # Fantasize each layers output.
        for i in range(self.num_layers):
            if i == 0:
                x = self.g(i)(x_initial)
            else:
                x = self.g(i)(x)
            x = torch.sigmoid(x)
            x = self.layer_output(x, self.training)
            results.append(x)

        return results

    def run_sleep(self, x, sample=None):
        batch_size = x.size(0)
        recognition_loss = []

        # We do not use the input `x`, rather we use the bias.
        if sample is not None:
            generation_bias_output = sample
        else:
            bias = self.g_bias.view(1, -1)
            x = torch.sigmoid(bias)
            x = x.expand(batch_size, self.g_bias.size(0))
            x = self.layer_output(x, self.training)
            generation_bias_output = x

        # Fantasize each layers output.
        generative_outputs = self._run_sleep_generation(generation_bias_output)

        # Run recognition layers to predict fantasies.
        recognition_loss = self._run_sleep_recognition(generation_bias_output, generative_outputs)

        return recognition_loss, generation_bias_output, generative_outputs
        
    def forward(self, x):
        x = torch.round(x)

        wake_out = self.run_wake(x)
        sleep_out = self.run_sleep(x)

        return wake_out + sleep_out
    
    def loss_function(self, 
                      rec_out, gen_bias_loss, gen_loss,   # awake forward()
                      rec_loss, gen_bias_out, gen_out):   # sleep forward()
        total_loss = gen_bias_loss
        for loss in (gen_loss + rec_loss):
            total_loss += loss
        
        return total_loss
    
    def sample(self, num_samples):
        fake_x = torch.zeros(num_samples)
        fantasy = self.run_sleep(fake_x)
        return fantasy[2][-1]

    def reconstruct(self, x):
        outputs = self.run_wake(x)
        sample = outputs[0][-1]
        results = self.run_sleep(x, sample)
        return results[2][-1]


class HM_color(nn.Module):
    def __init__(self, layers=None):
        super(HM_color, self).__init__()

        if layers is None:
            layers = [38804, 2048, 128, 32]

        self.rgb_models = [
            HM_bw(layers),
            HM_bw(layers),
            HM_bw(layers),
        ]

        self.params = ParameterList()
        for model in self.rgb_models:
            self.params.extend(model.parameters())
    

    def forward(self, x):
        """
        x must have shape N x C x H x W
        """
        x = torch.round(x)
        flat_dim = x.shape[-1] * x.shape[-2]
        color_layers = [x[:,i].reshape(-1, flat_dim) for i in range(3)]
        outputs = [model.forward(layer) for model, layer in zip(self.rgb_models, color_layers)]
        return outputs

    
    def loss_function(self, *fwd_outputs):
        losses = [model.loss_function(*output) for model, output in zip(self.rgb_models, fwd_outputs)]
        return sum(losses)

    
    def sample(self, num_samples):
        fake_x = torch.zeros(num_samples)
        fantasies = [model.run_sleep(fake_x)[2][-1] for model in self.rgb_models]
        return torch.stack(fantasies, dim=-1)


    def reconstruct(self, x):
        x = torch.round(x)
        flat_dim = x.shape[-1] * x.shape[-2]
        color_layers = [x[:,i].reshape(-1, flat_dim) for i in range(3)]
        images = [model.reconstruct(layer) for model, layer in zip(self.rgb_models, color_layers)]

        return torch.stack(images, dim=-1)


# <codecell>
class CelebADataset(Dataset):
    def __init__(self, im_path, total=TOTAL_IMAGES):
        self.im_path = im_path
        self.total = total

    def __getitem__(self, idx):
        name = str(idx + 1).zfill(6) + '.jpg'
        target_path = self.im_path / name

        # im = plt.imread(target_path).reshape(-1, *IM_DIMS)
        im = plt.imread(target_path).transpose((2, 0, 1))
        im = im.astype('float32') / 255
        return torch.from_numpy(im)

    def __len__(self):
        return self.total


def build_datasets(im_path: Path, total=TOTAL_IMAGES, train_test_split=0.01, seed=53110) -> (Dataset, Dataset):
    if type(im_path) == str:
        im_path = Path(im_path)

    ds = CelebADataset(im_path, total)
    total = len(ds)

    num_test = int(total * train_test_split)
    num_train = total - num_test

    test_ds, train_ds = random_split(ds, (num_test, num_train), generator=torch.Generator().manual_seed(seed))
    return train_ds, test_ds

# class MNIST(Dataset):
#     def __init__(self, mnist_path):
#         mnist = loadmat(mnist_path)
#         first = np.float32(mnist['trainX']) / 255
#         second = np.float32(mnist['testX']) / 255
#         self.data = np.concatenate((first, second))

#     def __getitem__(self, idx):
#         return torch.from_numpy(self.data[idx])

#     def __len__(self):
#         return self.data.shape[0]


# def build_datasets(im_path: Path, train_test_split=0.01, seed=53110) -> (Dataset, Dataset):
#     if type(im_path) == str:
#         im_path = Path(im_path)

#     ds = MNIST(im_path)
#     total = len(ds)

#     num_test = int(total * train_test_split)
#     num_train = total - num_test

#     test_ds, train_ds = random_split(ds, (num_test, num_train), generator=torch.Generator().manual_seed(seed))
#     return train_ds, test_ds

# class CelebASingleDataset(Dataset):
#     def __init__(self, im_path, channel_idx=0, total=TOTAL_IMAGES):
#         self.im_path = im_path
#         self.total = total
#         self.idx = channel_idx

#     def __getitem__(self, idx):
#         name = str(idx + 1).zfill(6) + '.jpg'
#         target_path = self.im_path / name

#         # im = plt.imread(target_path).reshape(-1, *IM_DIMS)
#         im = plt.imread(target_path).transpose((2, 0, 1))
#         im = im.astype('float32') / 255
#         return torch.from_numpy(im[self.idx].flatten())
#         # return torch.from_numpy(im)

#     def __len__(self):
#         return self.total

# def build_datasets(im_path: Path, total=TOTAL_IMAGES, train_test_split=0.01, seed=53110) -> (Dataset, Dataset):
#     if type(im_path) == str:
#         im_path = Path(im_path)

#     ds = CelebASingleDataset(im_path, total=total)
#     total = len(ds)

#     num_test = int(total * train_test_split)
#     num_train = total - num_test

#     test_ds, train_ds = random_split(ds, (num_test, num_train), generator=torch.Generator().manual_seed(seed))
#     return train_ds, test_ds


test_ds, train_ds = build_datasets(IM_PATH)

# <codecell>
hm = HM(color=True)
ckpt = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
hm.load_state_dict(ckpt['model_state_dict'])

hm.eval()

# <codecell>
# idx = 5
# samp_im = test_ds[idx].reshape(218, 178, 3)

# plt.imshow(samp_im)
# plt.show()

# <codecell>
idx = 15
samp = test_ds[idx].unsqueeze(0)
print(samp.shape)

with torch.no_grad():
    reco = hm.reconstruct(samp)

    print(torch.squeeze(reco).shape)
    print(torch.squeeze(samp).shape)

    reco_im = torch.squeeze(reco).reshape(218, 178, 3)
    samp_im = torch.squeeze(samp).permute(1, 2, 0)

plt.imshow(samp_im)
plt.show()
plt.imshow(reco_im)
plt.show()

# <codecell>
samp = [test_ds[i] for i in range(5)]   # index slices won't work on ds
samp = np.stack(samp)
samp = torch.from_numpy(samp)

with torch.no_grad():
    reco = hm.reconstruct(samp)

    reco_im = torch.squeeze(reco).reshape(-1, 218, 178, 3)
    samp_im = torch.squeeze(samp).permute(0, 2, 3, 1)

combined = np.empty((reco_im.shape[0] + samp_im.shape[0], 218, 178, 3))
combined[0::2] = samp_im
combined[1::2] = reco_im

fig = plt.figure(figsize=(10, 10))
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(5, 2),
                 axes_pad=0.1,
                 )

for ax, im in zip(grid, combined):
    ax.imshow(im)

fig.suptitle('HM reconstructions')
# plt.show()
plt.savefig('image/hm_bin_full_reco.png')


# <codecell>
with torch.no_grad():
    samp = hm.sample(25)
    samp_im = torch.squeeze(samp).reshape(25, 218, 178, 3)

fig = plt.figure(figsize=(10, 10))
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(5, 5),  # creates 2x2 grid of axes
                 axes_pad=0.1,  # pad between axes in inch.
                 )

for ax, im in zip(grid, samp_im):
    ax.imshow(im)

fig.suptitle('Sample faces drawn from HM')
# plt.show()
plt.savefig('image/hm_bin_full_sample.png')
