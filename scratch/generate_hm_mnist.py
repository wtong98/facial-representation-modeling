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
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, Dataset, random_split

# IM_DIMS = (178, 218)
# TOTAL_IMAGES = 202599
MODEL_PATH = Path('../save/hm/final.pt')
# DATA_PATH = Path('../data/')
# IM_PATH = DATA_PATH / 'img'
IM_PATH = '../mnist.mat'

train_test_split = 0.01

# <codecell>

class SmoothUnit(nn.Module):
    def __init__(self, in_feat: int, out_feat: int):
        super(SmoothUnit, self).__init__()
        self.linear = nn.Linear(in_feat, out_feat)
        self.log_var = Parameter(torch.FloatTensor(out_feat))
        self.log_var.data.uniform_(-1, 1)
    
    def forward(self, x):
        mu = self.linear(x) 
        logvar = self.log_var.expand(mu.shape[0], -1)
        var = torch.exp(logvar)
        return torch.normal(mu, var).detach(), (mu, logvar)


class HM_bw(nn.Module):

    def __init__(self, layers):
        super(HM_bw, self).__init__()
        if layers == None:
            self.layers = [784, 256, 64, 32]
        else:
            self.layers = layers

        self.num_layers = len(self.layers) - 1

        # recognition layers
        for ii, i in enumerate(range(self.num_layers)):
            self.set_r(ii, SmoothUnit(self.layers[i], self.layers[i+1]))

        # generation layers
        for ii, i in enumerate(range(self.num_layers)[::-1]):
            self.set_g(ii, SmoothUnit(self.layers[i+1], self.layers[i]))

        self.g_bias = Parameter(torch.FloatTensor(self.layers[-1]))
        self.g_bias_logvar = Parameter(torch.FloatTensor(self.layers[-1]))
        self.reset_parameters()


    def reset_parameters(self):
        self.g_bias.data.uniform_(-1, 1)
        self.g_bias_logvar.data.uniform_(-1, 1)


    def r(self, i):
        return getattr(self, "recognition_{}".format(i))

    def set_r(self, i, layer):
        setattr(self, "recognition_{}".format(i), layer)

    def g(self, i):
        return getattr(self, "generation_{}".format(i))

    def set_g(self, i, layer):
        setattr(self, "generation_{}".format(i), layer)
    

    def cross_entropy(self, source_params, target_params):
        mu_p, logvar_p = target_params
        mu_q, logvar_q = source_params

        ce = np.log(2 * np.pi * np.e) + logvar_p \
             + torch.exp(logvar_p - logvar_q) \
             + (torch.pow(mu_p - mu_q, 2) / torch.exp(logvar_q))  - 1 \
             + logvar_q - logvar_p
        ce = 0.5 * ce

        batch_size = mu_p.shape[0]
        return torch.sum(ce) / batch_size


    def _run_wake_recognition(self, x):
        results = []

        # Run recognition layers, saving stochastic outputs.
        for i in range(self.num_layers):
            x, params = self.r(i)(x)
            results.append((x, params))

        return results


    # TODO: observe logvar_p with default logvar_q = 0
    #       (maybe it should be forced to just mse)
    def _run_wake_generation(self, x_original, recognition_outputs, default_logvar=0):
        results = []

        # Run generative layers, predicting the input to each layer.
        for i in range(self.num_layers):
            if i == self.num_layers - 1:
                logvar = torch.zeros(x_original.shape) + default_logvar
                params_target = (x_original, default_logvar)
            else:
                _, params_target = recognition_outputs[-(i+2)]

            x_input, _ = recognition_outputs[-(i+1)]
            _, params = self.g(i)(x_input)
            results.append(self.cross_entropy(params, params_target))

        return results


    def run_wake(self, x):
        x_first = x
        batch_size = x.size(0)

        # Run Recognition Net.
        recognition_outputs = self._run_wake_recognition(x)

        # Fit the bias to the final layer.
        _, params = recognition_outputs[-1]
        g_params = (self.g_bias.expand(batch_size, -1), self.g_bias_logvar.expand(batch_size, -1))
        generation_bias_loss = self.cross_entropy(g_params, params)

        # Run Generation Net.
        generation_loss = self._run_wake_generation(x_first, recognition_outputs)
        return recognition_outputs, generation_bias_loss, generation_loss


    def _run_sleep_recognition(self, params_original, generative_outputs):
        results = []

        for i in range(self.num_layers):
            if i == self.num_layers - 1:
                params_target = params_original
            else:
                _, params_target = generative_outputs[-(i+2)]

            x_input, _ = generative_outputs[-(i+1)]
            _, params = self.r(i)(x_input)
            results.append(self.cross_entropy(params, params_target))

        return results


    def _run_sleep_generation(self, x):
        results = []

        # Fantasize each layers output.
        for i in range(self.num_layers):
            x, params = self.g(i)(x)
            results.append((x, params))

        return results


    # sample should be params tuple (mu, logvar)
    def run_sleep(self, x, sample=None):
        batch_size = x.shape[0]
        recognition_loss = []

        if sample != None:
            bias, logvar = sample
        else:
            bias = self.g_bias.view(1, -1)
            bias = bias.expand(batch_size, self.g_bias.size(0))
            logvar = self.g_bias_logvar.view(1, -1)
            logvar = logvar.expand(batch_size, self.g_bias_logvar.size(0))

        var = torch.exp(logvar)
        
        generation_bias_output = torch.normal(bias, var).detach()

        generative_outputs = self._run_sleep_generation(generation_bias_output)
        recognition_loss = self._run_sleep_recognition((bias, logvar), generative_outputs)
        return recognition_loss, generation_bias_output, generative_outputs
        

    def forward(self, x):
        wake_out = self.run_wake(x)
        sleep_out = self.run_sleep(x)

        return wake_out + sleep_out
    

    def loss_function(self, 
                      rec_out, gen_bias_loss, gen_loss,   # wake forward()
                      rec_loss, gen_bias_out, gen_out):   # sleep forward()
        total_loss = gen_bias_loss
        for loss in (gen_loss + rec_loss):
            total_loss += loss
        
        return total_loss

    
    def sample(self, num_samples):
        fake_x = torch.zeros(num_samples)
        fantasy = self.run_sleep(fake_x)
        return fantasy[2][-1][0]


    def reconstruct(self, x):
        outputs = self.run_wake(x)
        sample = outputs[0][-1][1]
        results = self.run_sleep(x, sample)
        return results[2][-1][0]

# class HM(nn.Module):

#     def __init__(self):
#         super(HM, self).__init__()

#         self._awake = True
#         # self.layers = [116412, 16384, 2048, 256, 32]
#         self.layers = [784, 256, 64, 32]
#         self.num_layers = len(self.layers) - 1

#         # recognition layers
#         for ii, i in enumerate(range(self.num_layers)):
#             self.set_r(ii, nn.Linear(self.layers[i], self.layers[i+1]))

#         # generation layers
#         for ii, i in enumerate(range(self.num_layers)[::-1]):
#             self.set_g(ii, nn.Linear(self.layers[i+1], self.layers[i]))

#         self.g_bias = Parameter(torch.FloatTensor(self.layers[-1]))

#         self.reset_parameters()

#     def reset_parameters(self):
#         self.g_bias.data.uniform_(-1, 1)

#     def r(self, i):
#         return getattr(self, "recognition_{}".format(i))

#     def set_r(self, i, layer):
#         setattr(self, "recognition_{}".format(i), layer)

#     def g(self, i):
#         return getattr(self, "generation_{}".format(i))

#     def set_g(self, i, layer):
#         setattr(self, "generation_{}".format(i), layer)

#     def wake(self):
#         self._awake = True

#     def sleep(self):
#         self._awake = False

#     def layer_output(self, x, training=True):
#         """
#         If training, treat x as bernoulli distribution and sample output,
#         otherwise simply round x, giving binary output in either case.
#         """
#         # if training:
#         #     out = torch.bernoulli(x).detach()
#         # else:
#         #     out = torch.round(x)
#         out = torch.bernoulli(x).detach()
#         return out

#     def _run_wake_recognition(self, x):
#         results = []

#         # Run recognition layers, saving stochastic outputs.
#         for i in range(self.num_layers):
#             x = self.r(i)(x)
#             x = F.sigmoid(x)
#             x = self.layer_output(x, self.training)
#             results.append(x)

#         return results

#     def _run_wake_generation(self, x_original, recognition_outputs):
#         results = []

#         # Run generative layers, predicting the input to each layer.
#         for i in range(self.num_layers):
#             x_input = recognition_outputs[-(i+1)]
#             if i == self.num_layers - 1:
#                 x_target = x_original
#             else:
#                 x_target = recognition_outputs[-(i+2)]
#             x = self.g(i)(x_input)
#             x = F.sigmoid(x)
#             results.append(nn.BCELoss()(x, x_target))

#         return results

#     def run_wake(self, x):
#         x_first = x
#         batch_size = x.size(0)

#         # Run Recognition Net.
#         recognition_outputs = self._run_wake_recognition(x)

#         # Fit the bias to the final layer.
#         x_last = recognition_outputs[-1]
#         x = self.g_bias.view(1, -1).expand(batch_size, self.g_bias.size(0))
#         x = F.sigmoid(x)
#         generation_bias_loss = nn.BCELoss()(x, x_last)

#         # Run Generation Net.
#         generation_loss = self._run_wake_generation(x_first, recognition_outputs)

#         return recognition_outputs, generation_bias_loss, generation_loss

#     def _run_sleep_recognition(self, x_initial, generative_outputs):
#         results = []

#         # Run recognition layers to predict fantasies.
#         for i in range(self.num_layers):
#             x_input = generative_outputs[-(i+1)]
#             if i == self.num_layers - 1:
#                 x_target = x_initial
#             else:
#                 x_target = generative_outputs[-(i+2)]
#             x = self.r(i)(x_input)
#             x = F.sigmoid(x)
#             results.append(nn.BCELoss()(x, x_target))

#         return results

#     def _run_sleep_generation(self, x_initial):
#         results = []

#         # Fantasize each layers output.
#         for i in range(self.num_layers):
#             if i == 0:
#                 x = self.g(i)(x_initial)
#             else:
#                 x = self.g(i)(x)
#             x = F.sigmoid(x)
#             x = self.layer_output(x, self.training)
#             results.append(x)

#         return results

#     def run_sleep(self, x):
#         batch_size = x.size(0)
#         recognition_loss = []

#         # We do not use the input `x`, rather we use the bias.
#         bias = self.g_bias.view(1, -1)
#         print('BIAS', bias)
#         x = F.sigmoid(bias)
#         print('BIAS_SIG', x)
#         x = x.expand(batch_size, self.g_bias.size(0))
#         x = self.layer_output(x, self.training)
#         generation_bias_output = x

#         # Fantasize each layers output.
#         generative_outputs = self._run_sleep_generation(generation_bias_output)

#         # Run recognition layers to predict fantasies.
#         recognition_loss = self._run_sleep_recognition(generation_bias_output, generative_outputs)

#         return recognition_loss, generation_bias_output, generative_outputs
        
#     def forward(self, x):
#         x = torch.round(x)

#         self.wake()
#         wake_out = self.run_wake(x)

#         self.sleep()
#         sleep_out = self.run_sleep(x)

#         return wake_out + sleep_out
    
#     def loss_function(self, 
#                       rec_out, gen_bias_loss, gen_loss,   # awake forward()
#                       rec_loss, gen_bias_out, gen_out):   # sleep forward()
#         total_loss = gen_bias_loss
#         for loss in (gen_loss + rec_loss):
#             total_loss += loss
        
#         # TODO: quick fix for accomdate vae
#         return {'loss': total_loss, 'kld': 0, 'mse': 0}

#     def sample(self, num_samples):
#         fake_x = torch.zeros(num_samples)
#         fantasy = self.run_sleep(fake_x)
#         return fantasy[2][-1]

#     def reconstruct(self, x):
#         results = self.forward(x)
        # return results[5][-1]


# <codecell>
class MNIST(Dataset):
    def __init__(self, mnist_path):
        mnist = loadmat(mnist_path)
        first = np.float32(mnist['trainX']) / 255
        second = np.float32(mnist['testX']) / 255
        self.data = np.concatenate((first, second))

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx])

    def __len__(self):
        return self.data.shape[0]


def build_datasets(im_path: Path, train_test_split=0.01, seed=53110) -> (Dataset, Dataset):
    if type(im_path) == str:
        im_path = Path(im_path)

    ds = MNIST(im_path)
    total = len(ds)

    num_test = int(total * train_test_split)
    num_train = total - num_test

    test_ds, train_ds = random_split(ds, (num_test, num_train), generator=torch.Generator().manual_seed(seed))
    return train_ds, test_ds

test_ds, train_ds = build_datasets(IM_PATH)

# <codecell>
hm = HM_bw(layers=None)
ckpt = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
hm.load_state_dict(ckpt['model_state_dict'])

hm.eval()

# <codecell>
# idx = 5
# samp_im = test_ds[idx].reshape(28, 28)

# plt.imshow(samp_im)
# plt.show()

# <codecell>
idx = 15
samp = test_ds[idx].unsqueeze(0)
print(samp.shape)

with torch.no_grad():
    reco = hm.reconstruct(samp)

    reco_im = torch.squeeze(reco).reshape(28, 28)
    samp_im = torch.squeeze(samp).reshape(28, 28)

plt.imshow(samp_im)
plt.show()
plt.imshow(reco_im)
plt.show()

# <codecell>
with torch.no_grad():
    samp = hm.sample(25)
    samp_im = torch.squeeze(samp).reshape(25, 28, 28)

fig = plt.figure(figsize=(10, 10))
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(5, 5),  # creates 2x2 grid of axes
                 axes_pad=0.1,  # pad between axes in inch.
                 )

for ax, im in zip(grid, samp_im):
    ax.imshow(im)

fig.suptitle('Sample faces drawn from HM')
plt.show()

# TODO: debug same image problem