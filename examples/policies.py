#!/usr/bin/env python3

import math

import cherry as ch
import torch as th
from torch import nn
from torch.distributions import Normal, Categorical

EPSILON = 1e-6


def linear_init(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        module.bias.data.zero_()
    return module


class LinearValue(nn.Module):

    def __init__(self, input_size, reg=1e-5):
        super(LinearValue, self).__init__()
        self.linear = nn.Linear(2 * input_size + 4, 1, bias=False)
        self.reg = reg

    def _features(self, states):
        length = states.size(0)
        ones = th.ones(length, 1)
        al = th.arange(length, dtype=th.float32).view(-1, 1) / 100.0
        return th.cat([states, states ** 2, al, al ** 2, al ** 3, ones], dim=1)

    def fit(self, states, returns):
        features = self._features(states)
        reg = self.reg * th.eye(features.size(1))
        A = features.t() @ features + reg
        b = features.t() @ returns
        coeffs, _ = th.gels(b, A)
        self.linear.weight.data = coeffs.data.t()

    def forward(self, states):
        features = self._features(states)
        return self.linear(features)


class DiagNormalPolicy(nn.Module):

    def __init__(self, input_size, output_size, hiddens=None):
        super(DiagNormalPolicy, self).__init__()
        if hiddens is None:
            hiddens = [100, 100]
        layers = [linear_init(nn.Linear(input_size, hiddens[0])), nn.ReLU()]
        for i, o in zip(hiddens[:-1], hiddens[1:]):
            layers.append(linear_init(nn.Linear(i, o)))
            layers.append(nn.ReLU())
        layers.append(linear_init(nn.Linear(hiddens[-1], output_size)))
        self.mean = nn.Sequential(*layers)
        self.sigma = nn.Parameter(th.Tensor(output_size))
        self.sigma.data.fill_(math.log(1))

    def forward(self, state):
        loc = self.mean(state)
        scale = th.exp(th.clamp(self.sigma, min=math.log(EPSILON)))
        density = Normal(loc=loc, scale=scale)
        action = density.sample()
        log_prob = density.log_prob(action).mean(dim=1,
                                                 keepdim=True).detach()
        return action, {'density': density, 'log_prob': log_prob}


class CategoricalPolicy(nn.Module):

    def __init__(self, input_size, output_size, hiddens=None):
        super(CategoricalPolicy, self).__init__()
        if hiddens is None:
            hiddens = [100, 100]
        layers = [linear_init(nn.Linear(input_size, hiddens[0])), nn.ReLU()]
        for i, o in zip(hiddens[:-1], hiddens[1:]):
            layers.append(linear_init(nn.Linear(i, o)))
            layers.append(nn.ReLU())
        layers.append(linear_init(nn.Linear(hiddens[-1], output_size)))
        self.mean = nn.Sequential(*layers)
        self.input_size = input_size

    def forward(self, state):
        state = ch.onehot(state, dim=self.input_size)
        loc = self.mean(state)
        density = Categorical(logits=loc)
        action = density.sample()
        log_prob = density.log_prob(action).mean().view(-1, 1).detach()
        return action, {'density': density, 'log_prob': log_prob}
