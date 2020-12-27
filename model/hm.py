"""
A simple Helmholtz machine model for learning faces, using real-valued units.
Adapated from: https://github.com/mrdrozdov/pytorch-machines/blob/master/helmholtz.py

author: William Tong
date: 12/1/2020
"""

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import ParameterList, functional as F
from torch.nn.parameter import Parameter


def HM(color=True, layers=None):
    # if color:
    #     return HM_color(layers)
    # else:
    #     return HM_bw(layers)
    return HM_bw(layers)


class SmoothUnit(nn.Module):
    def __init__(self, in_feat: int, out_feat: int):
        super(SmoothUnit, self).__init__()
        self.linear = nn.Linear(in_feat, out_feat)
        self.log_var = Parameter(torch.FloatTensor(out_feat))
        # self.log_var.data.uniform_(-1, 1)
        self.log_var.data.fill_(-3)

        self.mu = None # for debugging
    
    def forward(self, x):
        mu = self.linear(x) 
        self.mu = mu
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
        self.awake = True

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
        # self.g_bias_logvar.data.uniform_(-1, 1)
        self.g_bias_logvar.data.fill_(-3)


    def r(self, i):
        return getattr(self, "recognition_{}".format(i))

    def set_r(self, i, layer):
        setattr(self, "recognition_{}".format(i), layer)

    def g(self, i):
        return getattr(self, "generation_{}".format(i))

    def set_g(self, i, layer):
        setattr(self, "generation_{}".format(i), layer)
    

    def elbo(self, sample, samp_params, params):
        mu_p, logvar_p = params
        mu_q, logvar_q = samp_params
        # logvar_q = torch.zeros(logvar_q.shape) - 1

        mse = torch.pow(sample - mu_p, 2)

        kld = torch.exp(logvar_p - logvar_q) \
             + (torch.pow(mu_p - mu_q, 2) / torch.exp(logvar_q))  - 1 \
             + logvar_q - logvar_p

        loss = 0.5 * mse + 0 * kld
        batch_size = sample.shape[0]
        return torch.sum(loss) / batch_size
    

    def nll(self, sample, params):
        mu, logvar = params

        # nll = np.log(2 * np.pi * np.e) + logvar \
        #      + (torch.pow(sample - mu, 2) / torch.exp(logvar))
        # nll = 0.5 * nll

        # ent = np.log(2 * np.pi * np.e) + torch.sum(logvar)
        # f1 = torch.sum(torch.pow(sample - mu, 2) / torch.exp(logvar))

        # print('ENT:', ent)
        # print('F1:', f1)
        mse = torch.pow(sample - mu, 2)
        nll = mse

        batch_size = sample.shape[0]
        return torch.sum(nll) / batch_size
    

    def cross_entropy(self, source_params, target_params, mse_weight=0):
        mu_p, logvar_p = target_params
        mu_q, logvar_q = source_params

        mse = torch.pow(mu_p - mu_q, 2)
        # ce = mse

        # ce = np.log(2 * np.pi * np.e) + logvar_p \
        #      + (torch.pow(mu_p - mu_q, 2) / torch.exp(logvar_p))
        # ce = 0.5 * ce

        ce = np.log(2 * np.pi * np.e) + logvar_p \
             + torch.exp(logvar_p - logvar_q) \
             + (torch.pow(mu_p - mu_q, 2) / torch.exp(logvar_q))  - 1 \
             + logvar_q - logvar_p
        ce = 0.5 * ce

        # ce = torch.exp(logvar_p - logvar_q) \
        #      + (torch.pow(mu_p - mu_q, 2) / torch.exp(logvar_q))  - 1 \
        #      + logvar_q - logvar_p
        # # ce = 0.5 * ce + mse_weight * mse
        # ce = 0.5 * ce

        # ent = np.log(2 * np.pi * np.e) + torch.sum(logvar_p)
        # f1 = torch.sum(torch.exp(logvar_p - logvar_q))
        # f2 = torch.sum((torch.pow(mu_p - mu_q, 2) / torch.exp(logvar_q))  - 1)
        # f3 = torch.sum(logvar_q - logvar_p)

        # print('ENT', ent)
        # print('F1', f1)
        # print('F2', f2)
        # print('F3', f3)

        batch_size = mu_p.shape[0]
        return torch.sum(ce) / batch_size


    def _run_wake_recognition(self, x):
        results = []

        # Run recognition layers, saving stochastic outputs.
        for i in range(self.num_layers):
            x, params = self.r(i)(x)
            results.append((x, params))

        return results


    # TODO: observe logvar_q with default logvar_p = 0
    #       (maybe it should be forced to just mse)
    def _run_wake_generation(self, x_original, recognition_outputs, default_logvar=-2):
        results = []

        # Run generative layers, predicting the input to each layer.
        for i in range(self.num_layers):
            if i == self.num_layers - 1:
                logvar = torch.zeros(x_original.shape) + default_logvar
                sample = x_original
                samp_params = (sample, logvar)
            else:
                sample, samp_params = recognition_outputs[-(i+2)]

            x_input, _ = recognition_outputs[-(i+1)]
            _, params = self.g(i)(x_input)
            # results.append(self.cross_entropy(params, params_target, mse_weight))
            results.append(self.elbo(sample, samp_params, params))

        return results


    def run_wake(self, x):
        x_first = x
        batch_size = x.size(0)

        # Run Recognition Net.
        recognition_outputs = self._run_wake_recognition(x)

        # Fit the bias to the final layer.
        sample, samp_params = recognition_outputs[-1]
        params = (self.g_bias.expand(batch_size, -1), self.g_bias_logvar.expand(batch_size, -1))
        # generation_bias_loss = self.cross_entropy(g_params, params)
        generation_bias_loss = self.elbo(sample, samp_params, params)

        # Run Generation Net.
        generation_loss = self._run_wake_generation(x_first, recognition_outputs)
        return recognition_outputs, generation_bias_loss, generation_loss


    def _run_sleep_recognition(self, x_original, params_original, generative_outputs):
        results = []

        for i in range(self.num_layers):
            if i == self.num_layers - 1:
                sample, samp_params = x_original, params_original
            else:
                sample, samp_params = generative_outputs[-(i+2)]

            x_input, _ = generative_outputs[-(i+1)]
            _, params = self.r(i)(x_input)
            results.append(self.elbo(sample, samp_params, params))

        return results


    def _run_sleep_generation(self, x):
        results = []

        # Fantasize each layers output.
        for i in range(self.num_layers):
            x, params = self.g(i)(x)
            results.append((x, params))

        return results


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
        g_params = (bias, logvar)

        generative_outputs = self._run_sleep_generation(generation_bias_output)
        recognition_loss = self._run_sleep_recognition(generation_bias_output, g_params, generative_outputs)
        return recognition_loss, generation_bias_output, generative_outputs
        

    # TODO: skips examples if training size is even
    # TODO: eval will not flip flop
    def forward(self, x):
        if self.awake:
            wake_out = self.run_wake(x)
            sleep_out = ([0], 0, [0])
        else:
            wake_out = ([0], 0, [0])
            sleep_out = self.run_sleep(x)
        
        self.awake = not self.awake
        # wake_out = self.run_wake(x)
        # sleep_out = self.run_sleep(x)

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


# class HM_color(nn.Module):
#     def __init__(self, layers=None):
#         super(HM_color, self).__init__()

#         if layers is None:
#             layers = [38804, 2048, 128, 32]

#         self.rgb_models = [
#             HM_bw(layers),
#             HM_bw(layers),
#             HM_bw(layers),
#         ]

#         self.params = ParameterList()
#         for model in self.rgb_models:
#             self.params.extend(model.parameters())
    

#     def forward(self, x):
#         """
#         x must have shape N x C x H x W
#         """
#         x = torch.round(x)
#         flat_dim = x.shape[-1] * x.shape[-2]
#         color_layers = [x[:,i].reshape(-1, flat_dim) for i in range(3)]
#         outputs = [model.forward(layer) for model, layer in zip(self.rgb_models, color_layers)]
#         return outputs

    
#     def loss_function(self, *fwd_outputs):
#         losses = [model.loss_function(*output) for model, output in zip(self.rgb_models, fwd_outputs)]
#         return sum(losses)

    
#     def sample(self, num_samples):
#         fake_x = torch.zeros(num_samples)
#         fantasies = [model.run_sleep(fake_x)[2][-1] for model in self.rgb_models]
#         return torch.stack(fantasies, dim=-1)


#     def reconstruct(self, x):
#         outputs = self.forward(x)
#         images = [result[5][-1] for result in outputs]
#         return torch.stack(images, dim=-1)
