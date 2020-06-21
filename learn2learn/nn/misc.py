#!/usr/bin/env python3

import torch


class Lambda(torch.nn.Module):
    """
    docs
    """

    def __init__(self, lmb):
        super(Lambda, self).__init__()
        self.lmb = lmb

    def forward(self, *args, **kwargs):
        return self.lmb(*args, **kwargs)


class Flatten(torch.nn.Module):
    """
    docs
    """

    def forward(self, x):
        return x.view(x.size(0), -1)


class Scale(torch.nn.Module):
    """
    docs
    """

    def __init__(self, input_size, output_size, lr=1.0):
        super(Scale, self).__init__()
        assert input_size == output_size, \
            'input_size and output_size must match.'
        lr = torch.ones(input_size)
        self.lr = torch.nn.Parameter(lr)

    def forward(self, x):
        return x * self.lr
