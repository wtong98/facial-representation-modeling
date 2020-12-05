"""
A simple Helmholtz machine model for learning faces.
Adapated from: https://github.com/mrdrozdov/pytorch-machines/blob/master/helmholtz.py

author: William Tong
date: 12/1/2020
"""

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import ParameterList, functional as F
from torch.nn.parameter import Parameter


def HM(color=True, layers=None):
    if color:
        return HM_color(layers)
    else:
        return HM_bw(layers)


class HM_bw(nn.Module):

    def __init__(self, layers=None):
        super(HM_bw, self).__init__()

        if layers is None:
            # self.layers = [116412, 16384, 2048, 256, 32]
            self.layers = [784, 256, 64, 32]
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

    def run_sleep(self, x):
        batch_size = x.size(0)
        recognition_loss = []

        # We do not use the input `x`, rather we use the bias.
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
        results = self.forward(x)
        return results[5][-1]


class HM_color(nn.Module):
    def __init__(self, layers=None):
        super(HM_color, self).__init__()

        if layers is None:
            layers = [38804, 4096, 1024, 256, 64]

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
        outputs = self.forward(x)
        images = [result[5][-1] for result in outputs]
        return torch.stack(images, dim=-1)

    


