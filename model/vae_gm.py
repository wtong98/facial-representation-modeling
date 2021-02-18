"""
Describes variational autoencoder model for generating faces. Uses a 
Gaussian mixture prior, rather than a standard Gaussian.

author: William Tong (wlt2115@columbia.edu)
date: 11/5/2020
"""

import numpy as np

import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.container import ParameterList
from torch.nn.parameter import Parameter

class VAE(nn.Module):

    def __init__(self):
        super(VAE, self).__init__()

        self.latent_x = 40      # total hidden representation
        self.latent_w = 20      # hidden representation per cluster
        self.clusters = 10      # number of discrete clusters learned by the model
        self.beta_width = 100   # width of hidden layer transforming w_k --> x
        self.mc_samples = 5     # number of Monte Carlo samples to compute at each step

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

        center_size = 128*14*12
        self.fc_mu_x = nn.Linear(center_size, self.latent_x)
        self.fc_var_x = nn.Linear(center_size, self.latent_x)
        self.fc_mu_w = nn.Linear(center_size, self.latent_w)
        self.fc_var_w = nn.Linear(center_size, self.latent_w)

        self.decoder_input = nn.Linear(self.latent_x, center_size)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=5, 
                               stride=2, padding=2, output_padding=(1,0)),
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

        self.fc_mu_cluster = [nn.Sequential(
            nn.Linear(self.latent_w, self.beta_width),
            nn.Linear(self.beta_width, self.latent_x),
            nn.Tanh()
        ) for _ in range(self.clusters)]

        self.fc_var_cluster = [nn.Sequential(
            nn.Linear(self.latent_w, self.beta_width),
            nn.Linear(self.beta_width, self.latent_x),
            nn.Tanh()
        ) for _ in range(self.clusters)]

        self.params = ParameterList()
        for model in self.fc_mu_cluster + self.fc_var_cluster:
            self.params.extend(model.parameters())


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
        mu_x = self.fc_mu_x(result)
        log_var_x = self.fc_var_x(result)

        mu_w = self.fc_mu_w(result)
        log_var_w = self.fc_var_w(result)

        return ((mu_x, log_var_x), (mu_w, log_var_w))


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
        params_x, params_w = self.encode(data)

        samples = []
        for _ in range(self.mc_samples):
            sample_x = self.reparameterize(*params_x)
            sample_w = self.reparameterize(*params_w)

            reco = self.decode(sample_x)

            cluster_mu = [model(sample_w) for model in self.fc_mu_cluster]
            cluster_var = [model(sample_w) for model in self.fc_var_cluster]

            samples.append({
                'orig': data,
                'reco': reco,
                'sample_x': sample_x,
                'sample_w': sample_w,
                'params_x': params_x,
                'params_w': params_w,
                'cluster_mu': cluster_mu,
                'cluster_var': cluster_var
            })
    
        return samples


    # def loss_function(self, recons, data, mu, log_var, kld_weight=1) -> dict:
    # TODO: add KL loss tuning param
    # TODO: think through implications of batch size
    def loss_function(self, samples) -> dict:
        """
        Computes the VAE loss function.
        """
        reco_loss = self._reco_loss(samples)
        z_prior_loss = self._z_prior_loss(samples)
        w_prior_loss = self._w_prior_loss(samples)
        cond_prior_loss = self._cond_prior_loss(samples)
        total_loss = reco_loss + z_prior_loss + w_prior_loss + cond_prior_loss

        return {
            'loss': total_loss,
            'reco_loss': reco_loss,
            'z_prior_loss': z_prior_loss,
            'w_prior_loss': w_prior_loss,
            'cond_prior_loss': cond_prior_loss,
        }
    

    def _reco_loss(self, samples):
        total_loss = 0
        for samp in samples:
            orig = samp['orig']
            reco = samp['reco']
            _, logvar = samp['params_x']

            mse = torch.pow(orig - reco, 2)
            samp_loss = (1/2 * torch.log(2 * np.pi)) \
                        + logvar \
                        + (mse / (2 * logvar.exp()))
            total_loss += torch.sum(samp_loss)

        return total_loss / len(samples)


    def _z_prior_loss(self, samples, eps=1e-10):
        total_loss = 0
        for samp in samples:
            cluster_mu = samp['cluster_mu']
            cluster_var = samp['cluster_var']
            sample_x = samp['sample_x']

            z_post = self._z_pos(sample_x, cluster_mu, cluster_var)
            kld_terms = z_post * (torch.log(z_post + eps) + np.log(self.clusters))
            total_loss += torch.sum(kld_terms)
            
        return total_loss / len(samples)


    def _w_prior_loss(self, samples):
        total_loss = 0
        for samp in samples:
            params_w = samp['params_w']
            params_w_prior = (torch.zeros(self.latent_w), torch.ones(self.latent_w))
            total_loss += self._gauss_kld(params_w, params_w_prior)

        return total_loss / len(samples)
    

    def _cond_prior_loss(self, samples):
        total_loss = 0
        for samp in samples:
            params_x = samp['params_x']
            cluster_mu = samp['cluster_mu']
            cluster_var = samp['cluster_var']
            sample_x = samp['sample_x']

            z_post = self._z_post(sample_x, cluster_mu, cluster_var)

            for i, z_prob in enumerate(z_post):
                mu_c = cluster_mu[i]
                logvar_c = cluster_var[i]

                kld = self._gauss_kld(params_x, (mu_c, logvar_c))
                total_loss += z_prob * kld

        return total_loss / len(samples)


    def _z_post(self, sample_x, cluster_mu, cluster_var):  # assuming uniform prior on z
        full_var = [torch.exp(logvar) for logvar in cluster_var]
        dists = [MultivariateNormal(loc=mu, covariance_matrix=torch.diag(var)) 
                 for (mu, var) in zip(cluster_mu, full_var)]
        probs = torch.Tensor([torch.exp(dist.log_prob(sample_x)) for dist in dists])

        return probs / torch.sum(probs)
    

    def _gauss_kld(self, params_p, params_q):
        mu_p, logvar_p = params_p
        mu_q, logvar_q = params_q

        kld2 = (logvar_p.exp() / logvar_q.exp()) \
                + (torch.pow(mu_q - mu_p, 2) / logvar_q.exp()) \
                + 2 * (logvar_q - logvar_p) \
                - 1
        return 0.5 * kld2

    
    # TODO: let's train this beast! <--- STOPPED HERE
    def sample(self, num_samples : int, cluster_id : int) -> 'Tensor':
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """

        w = torch.randn(num_samples, self.latent_w)
        mu_x = self.fc_mu_cluster[cluster_id]
        logvar_x = self.fc_var_cluster[cluster_id]

        sample_x = self.reparameterize(mu_x, logvar_x)
        # first_tensor = next(self.parameters())
        # if first_tensor.is_cuda:
        #     z = z.to(torch.device('cuda'))

        samples = self.decode(sample_x)
        return samples
    

    def reconstruct(self, x: 'Tensor', **kwargs) -> 'Tensor':
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)['reco']

