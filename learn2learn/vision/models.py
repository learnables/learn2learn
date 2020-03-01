#!/usr/bin/env python3

"""
**Description**

A set of commonly used models for meta-learning vision tasks.
"""

import torch
from scipy.stats import truncnorm
from torch import nn


def truncated_normal_(tensor, mean=0.0, std=1.0):
    # PT doesn't have truncated normal.
    # https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/18
    values = truncnorm.rvs(-2, 2, size=tensor.shape)
    values = mean + std * values
    tensor.copy_(torch.from_numpy(values))
    return tensor


def fc_init_(module):
    if hasattr(module, 'weight') and module.weight is not None:
        truncated_normal_(module.weight.data, mean=0.0, std=0.01)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias.data, 0.0)
    return module


def maml_init_(module):
    nn.init.xavier_uniform_(module.weight.data, gain=1.0)
    nn.init.constant_(module.bias.data, 0.0)
    return module


class LinearBlock(nn.Module):

    def __init__(self, input_size, output_size):
        super(LinearBlock, self).__init__()
        self.relu = nn.ReLU()
        self.normalize = nn.BatchNorm1d(output_size,
                                        affine=True,
                                        momentum=0.999,
                                        eps=1e-3,
                                        track_running_stats=False,
                                        )
        self.linear = nn.Linear(input_size, output_size)
        fc_init_(self.linear)

    def forward(self, x):
        x = self.linear(x)
        x = self.normalize(x)
        x = self.relu(x)
        return x


class ConvBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 max_pool=True,
                 max_pool_factor=1.0):
        super(ConvBlock, self).__init__()
        stride = (int(2 * max_pool_factor), int(2 * max_pool_factor))
        if max_pool:
            self.max_pool = nn.MaxPool2d(kernel_size=stride,
                                         stride=stride,
                                         ceil_mode=False,
                                         )
            stride = (1, 1)
        else:
            self.max_pool = lambda x: x
        self.normalize = nn.BatchNorm2d(out_channels,
                                        affine=True,
                                        # eps=1e-3,
                                        # momentum=0.999,
                                        # track_running_stats=False,
                                        )
        nn.init.uniform_(self.normalize.weight)
        self.relu = nn.ReLU()

        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size,
                              stride=stride,
                              padding=1,
                              bias=True)
        maml_init_(self.conv)

    def forward(self, x):
        x = self.conv(x)
        x = self.normalize(x)
        x = self.relu(x)
        x = self.max_pool(x)
        return x


class ConvBase(nn.Sequential):

    # NOTE:
    #     Omniglot: hidden=64, channels=1, no max_pool
    #     MiniImagenet: hidden=32, channels=3, max_pool

    def __init__(self,
                 output_size,
                 hidden=64,
                 channels=1,
                 max_pool=False,
                 layers=4,
                 max_pool_factor=1.0):
        core = [ConvBlock(channels,
                          hidden,
                          (3, 3),
                          max_pool=max_pool,
                          max_pool_factor=max_pool_factor),
                ]
        for l in range(layers - 1):
            core.append(ConvBlock(hidden,
                                  hidden,
                                  kernel_size=(3, 3),
                                  max_pool=max_pool,
                                  max_pool_factor=max_pool_factor))
        super(ConvBase, self).__init__(*core)


class OmniglotFC(nn.Sequential):
    """

    [[Source]]()

    **Description**

    The fully-connected network used for Omniglot experiments, as described in Santoro et al, 2016.

    **References**

    1. Santoro et al. 2016. “Meta-Learning with Memory-Augmented Neural Networks.” ICML.

    **Arguments**

    * **input_size** (int) - The dimensionality of the input.
    * **output_size** (int) - The dimensionality of the output.
    * **sizes** (list, *optional*, default=None) - A list of hidden layer sizes.

    **Example**
    ~~~python
    net = OmniglotFC(input_size=28**2,
                     output_size=10,
                     sizes=[64, 64, 64])
    ~~~

    """

    def __init__(self, input_size, output_size, sizes=None):
        if sizes is None:
            sizes = [256, 128, 64, 64]
        layers = [LinearBlock(input_size, sizes[0]), ]
        for s_i, s_o in zip(sizes[:-1], sizes[1:]):
            layers.append(LinearBlock(s_i, s_o))
        layers.append(fc_init_(nn.Linear(sizes[-1], output_size)))
        super(OmniglotFC, self).__init__(*layers)
        self.input_size = input_size

    def forward(self, x):
        return super(OmniglotFC, self).forward(x.view(-1, self.input_size))


class OmniglotCNN(nn.Module):
    """

    [Source]()

    **Description**

    The convolutional network commonly used for Omniglot, as described by Finn et al, 2017.

    This network assumes inputs of shapes (1, 28, 28).

    **References**

    1. Finn et al. 2017. “Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks.” ICML.

    **Arguments**

    * **output_size** (int) - The dimensionality of the network's output.
    * **hidden_size** (int, *optional*, default=64) - The dimensionality of the hidden representation.
    * **layers** (int, *optional*, default=4) - The number of convolutional layers.

    **Example**
    ~~~python
    model = OmniglotCNN(output_size=20, hidden_size=128, layers=3)
    ~~~

    """

    def __init__(self, output_size=5, hidden_size=64, layers=4):
        super(OmniglotCNN, self).__init__()
        self.hidden_size = hidden_size
        self.base = ConvBase(output_size=hidden_size,
                             hidden=hidden_size,
                             channels=1,
                             max_pool=False,
                             layers=layers)
        self.linear = nn.Linear(hidden_size, output_size, bias=True)
        self.linear.weight.data.normal_()
        self.linear.bias.data.mul_(0.0)

    def forward(self, x):
        x = self.base(x.view(-1, 1, 28, 28))
        x = x.mean(dim=[2, 3])
        x = self.linear(x)
        return x


class MiniImagenetCNN(nn.Module):
    """

    [[Source]]()

    **Description**

    The convolutional network commonly used for MiniImagenet, as described by Ravi et Larochelle, 2017.

    This network assumes inputs of shapes (3, 84, 84).

    **References**

    1. Ravi and Larochelle. 2017. “Optimization as a Model for Few-Shot Learning.” ICLR.

    **Arguments**

    * **output_size** (int) - The dimensionality of the network's output.
    * **hidden_size** (int, *optional*, default=32) - The dimensionality of the hidden representation.
    * **layers** (int, *optional*, default=4) - The number of convolutional layers.

    **Example**
    ~~~python
    model = MiniImagenetCNN(output_size=20, hidden_size=128, layers=3)
    ~~~
    """

    def __init__(self, output_size, hidden_size=32, layers=4):
        super(MiniImagenetCNN, self).__init__()
        self.base = ConvBase(output_size=hidden_size,
                             hidden=hidden_size,
                             channels=3,
                             max_pool=True,
                             layers=layers,
                             max_pool_factor=4 // layers)
        self.linear = nn.Linear(25 * hidden_size, output_size, bias=True)
        maml_init_(self.linear)
        self.hidden_size = hidden_size

    def forward(self, x):
        x = self.base(x)
        x = self.linear(x.view(-1, 25 * self.hidden_size))
        return x
