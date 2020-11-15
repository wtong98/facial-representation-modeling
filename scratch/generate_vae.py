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
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, random_split

IM_DIMS = (178, 218)
TOTAL_IMAGES = 202599
MODEL_PATH = Path('vae_save/final.pt')
DATA_PATH = Path('../data/')
IM_PATH = DATA_PATH / 'img'

latent_dims = 40
train_test_split = 0.01

# <codecell>
class VAE(nn.Module):

    def __init__(self, latent_dims):
        super(VAE, self).__init__()

        self.latent_dims = latent_dims

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(num_features=16),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(),
        )

        center_size = 128*12*14
        self.fc_mu = nn.Linear(center_size, self.latent_dims)
        self.fc_var = nn.Linear(center_size, self.latent_dims)

        self.decoder_input = nn.Linear(self.latent_dims, center_size)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=5, 
                               stride=2, padding=2, output_padding=(0,1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=5, 
                               stride=2, padding=2, output_padding=0),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=5, 
                               stride=2, padding=2, output_padding=0),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=5,
                               stride=2, padding=2, output_padding=1),
            nn.Sigmoid()
        )

    def encode(self, data: 'Tensor') -> 'List[Tensor]':
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param data: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """

        result = self.encoder(data)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]


    def decode(self, z: 'Tensor') -> 'Tensor':
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """

        result = self.decoder_input(z)
        result = result.view(-1, 128, 12, 14)
        result = self.decoder(result)
        return result


    def reparameterize(self, mu: 'Tensor', logvar: 'Tensor') -> 'Tensor':
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    
    def forward(self, data: 'Tensor', **kwargs) -> 'List[Tensor]':
        mu, log_var = self.encode(data)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), data, mu, log_var]
    

    def loss_function(self, recons, data, mu, log_var, kld_weight=1) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """

        recons_loss = 0.5 * F.mse_loss(recons, data, reduction='sum')
        kld_loss = torch.sum(-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim = 1), dim = 0)
        loss = (recons_loss + kld_weight * kld_loss) / batch_size
        return {'loss': loss, 'mse':recons_loss, 'kld':-kld_loss}
    

    def sample(self, num_samples:int) -> 'Tensor':
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """

        z = torch.randn(num_samples, self.latent_dims).double()

        # TODO: for gpu power:
        # z = z.to(current_device)

        samples = self.decode(z)
        return samples
    

    def reconstruct(self, x: 'Tensor', **kwargs) -> 'Tensor':
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]



# <codecell>

# TODO: unify code better
def _idx_to_im(im_idx):
    name = str(im_idx + 1).zfill(6) + '.jpg'
    im_path = IM_PATH / name

    im = plt.imread(im_path).reshape(-1, *IM_DIMS)
    im = im.astype('double') / 255
    return torch.from_numpy(im)


class CelebADataset(Dataset):
    def __getitem__(self, idx):
        return _idx_to_im(idx)

    def __len__(self):
        return TOTAL_IMAGES

num_test = int(TOTAL_IMAGES * train_test_split)
num_train = TOTAL_IMAGES - num_test

ds = CelebADataset()
test_ds, train_ds = random_split(ds, (num_test, num_train))

# <codecell>
vae = VAE(latent_dims=latent_dims).double()
ckpt = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
vae.load_state_dict(ckpt['model_state_dict'])

vae.eval()

# <codecell>
idx = 5
samp_im = test_ds[idx].reshape(218, 178, 3)

plt.imshow(samp_im)
plt.show()

# <codecell>
idx = 5
samp = test_ds[idx].unsqueeze(0)
print(samp.shape)

with torch.no_grad():
    reco = vae.reconstruct(samp)

    reco_im = torch.squeeze(reco).reshape(218, 178, 3)
    samp_im = torch.squeeze(samp).reshape(218, 178, 3)

plt.imshow(samp_im)
plt.show()
plt.imshow(reco_im)
plt.show()

# <codecell>
with torch.no_grad():
    samp = vae.sample(25)
    samp_im = torch.squeeze(samp).reshape(25, 218, 178, 3)

fig = plt.figure(figsize=(10, 10))
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(5, 5),  # creates 2x2 grid of axes
                 axes_pad=0.1,  # pad between axes in inch.
                 )

for ax, im in zip(grid, samp_im):
    ax.imshow(im)

fig.suptitle('Sample faces drawn from VAE')
plt.show()
