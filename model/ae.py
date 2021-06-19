"""
Describes *autoencoder* model for generating faces

author: William Tong (wlt2115@columbia.edu)
date: 6/6/2020
"""

import torch
from torch import nn
from torch.nn import functional as F


class AE(nn.Module):
    # TODO: train and run

    # TODO: adapt to autoencoder arch
    def __init__(self, latent_dims=40):
        super(AE, self).__init__()

        self.latent_dims = latent_dims

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16,
                      kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(num_features=16),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels=16, out_channels=32,
                      kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels=64, out_channels=128,
                      kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(),
        )

        # TODO: center layer may be easiest way to control latent dims
        center_size = 128*14*12
        self.fc_center = nn.Linear(center_size, self.latent_dims)

        self.decoder_input = nn.Linear(self.latent_dims, center_size)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=5,
                               stride=2, padding=2, output_padding=(1, 0)),
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

        # def reinit(m):
        #     if isinstance(m, nn.Conv2d):
        #         m.reset_parameters()
        #         m.weight.data *= 0.5

        # self.encoder.apply(reinit)

    def encode(self, data: 'Tensor') -> 'List[Tensor]':
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param data: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """

        result = self.encoder(data)
        result = torch.flatten(result, start_dim=1)

        embedding = self.fc_center(result)

        return embedding

    def decode(self, z: 'Tensor') -> 'Tensor':
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """

        result = self.decoder_input(z)
        result = result.view(-1, 128, 14, 12)
        result = self.decoder(result)
        return result

    def forward(self, data: 'Tensor') -> 'List[Tensor]':
        embedding = self.encode(data)
        return [self.decode(embedding), data]

    def loss_function(self, samples) -> dict:
        """
        Computes the AE loss function.
        """
        recons, data = samples
        batch_size = recons.shape[0]

        loss = F.mse_loss(recons, data, reduction='sum') / batch_size
        return {'loss': loss}

    def sample(self, num_samples: int) -> 'Tensor':
        pass  # TODO: this could be fun

    def reconstruct(self, x: 'Tensor') -> 'Tensor':
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]

    def print_loss(self, loss):
        return "loss: {loss}".format(**loss)
