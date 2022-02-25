#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#

"""
CNN4 extended with Batch-Norm Running Statistics.
"""

import torch
import torch.nn.functional as F

from copy import deepcopy
from learn2learn.vision.models.cnn4 import maml_init_, fc_init_


class MetaBatchNormLayer(torch.nn.Module):
    """
    An extension of Pytorch's BatchNorm layer, with the Per-Step Batch Normalisation Running
    Statistics and Per-Step Batch Normalisation Weights and Biases improvements proposed in
    MAML++ by Antoniou et al. It is adapted from the original Pytorch implementation at
    https://github.com/AntreasAntoniou/HowToTrainYourMAMLPytorch,
    with heavy refactoring and a bug fix
    (https://github.com/AntreasAntoniou/HowToTrainYourMAMLPytorch/issues/42).
    """

    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        meta_batch_norm=True,
        adaptation_steps: int = 1,
    ):
        super(MetaBatchNormLayer, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.meta_batch_norm = meta_batch_norm
        self.num_features = num_features
        self.running_mean = torch.nn.Parameter(
            torch.zeros(adaptation_steps, num_features), requires_grad=False
        )
        self.running_var = torch.nn.Parameter(
            torch.ones(adaptation_steps, num_features), requires_grad=False
        )
        self.bias = torch.nn.Parameter(
            torch.zeros(adaptation_steps, num_features), requires_grad=True
        )
        self.weight = torch.nn.Parameter(
            torch.ones(adaptation_steps, num_features), requires_grad=True
        )
        self.backup_running_mean = torch.zeros(self.running_mean.shape)
        self.backup_running_var = torch.ones(self.running_var.shape)
        self.momentum = momentum

    def forward(
        self,
        input,
        step,
    ):
        """
        :param input: input data batch, size either can be any.
        :param step: The current inner loop step being taken. This is used when to learn per step params and
         collecting per step batch statistics.
        :return: The result of the batch norm operation.
        """
        assert (
            step < self.running_mean.shape[0]
        ), f"Running forward with step={step} when initialised with {self.running_mean.shape[0]} steps!"
        return F.batch_norm(
            input,
            self.running_mean[step],
            self.running_var[step],
            self.weight[step],
            self.bias[step],
            training=True,
            momentum=self.momentum,
            eps=self.eps,
        )

    def backup_stats(self):
        self.backup_running_mean.data = deepcopy(self.running_mean.data)
        self.backup_running_var.data = deepcopy(self.running_var.data)

    def restore_backup_stats(self):
        """
        Resets batch statistics to their backup values which are collected after each forward pass.
        """
        self.running_mean = torch.nn.Parameter(
            self.backup_running_mean, requires_grad=False
        )
        self.running_var = torch.nn.Parameter(
            self.backup_running_var, requires_grad=False
        )

    def extra_repr(self):
        return "{num_features}, eps={eps}, momentum={momentum}, affine={affine}".format(
            **self.__dict__
        )


class LinearBlock_BNRS(torch.nn.Module):
    def __init__(self, input_size, output_size, adaptation_steps):
        super(LinearBlock_BNRS, self).__init__()
        self.relu = torch.nn.ReLU()
        self.normalize = MetaBatchNormLayer(
            output_size,
            affine=True,
            momentum=0.999,
            eps=1e-3,
            adaptation_steps=adaptation_steps,
        )
        self.linear = torch.nn.Linear(input_size, output_size)
        fc_init_(self.linear)

    def forward(self, x, step):
        x = self.linear(x)
        x = self.normalize(x, step)
        x = self.relu(x)
        return x


class ConvBlock_BNRS(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        max_pool=True,
        max_pool_factor=1.0,
        adaptation_steps=1,
    ):
        super(ConvBlock_BNRS, self).__init__()
        stride = (int(2 * max_pool_factor), int(2 * max_pool_factor))
        if max_pool:
            self.max_pool = torch.nn.MaxPool2d(
                kernel_size=stride,
                stride=stride,
                ceil_mode=False,
            )
            stride = (1, 1)
        else:
            self.max_pool = lambda x: x
        self.normalize = MetaBatchNormLayer(
            out_channels,
            affine=True,
            adaptation_steps=adaptation_steps,
            # eps=1e-3,
            # momentum=0.999,
        )
        torch.nn.init.uniform_(self.normalize.weight)
        self.relu = torch.nn.ReLU()

        self.conv = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=1,
            bias=True,
        )
        maml_init_(self.conv)

    def forward(self, x, step):
        x = self.conv(x)
        x = self.normalize(x, step)
        x = self.relu(x)
        x = self.max_pool(x)
        return x


class ConvBase_BNRS(torch.nn.Sequential):

    # NOTE:
    #     Omniglot: hidden=64, channels=1, no max_pool
    #     MiniImagenet: hidden=32, channels=3, max_pool

    def __init__(
        self, hidden=64, channels=1, max_pool=False, layers=4, max_pool_factor=1.0,
        adaptation_steps=1
    ):
        core = [
            ConvBlock_BNRS(
                channels,
                hidden,
                (3, 3),
                max_pool=max_pool,
                max_pool_factor=max_pool_factor,
                adaptation_steps=adaptation_steps
            ),
        ]
        for _ in range(layers - 1):
            core.append(
                ConvBlock_BNRS(
                    hidden,
                    hidden,
                    kernel_size=(3, 3),
                    max_pool=max_pool,
                    max_pool_factor=max_pool_factor,
                    adaptation_steps=adaptation_steps
                )
            )
        super(ConvBase_BNRS, self).__init__(*core)

    def forward(self, x, step):
        for module in self:
            x = module(x, step)
        return x


class CNN4Backbone_BNRS(ConvBase_BNRS):
    def __init__(
        self,
        hidden_size=64,
        layers=4,
        channels=3,
        max_pool=True,
        max_pool_factor=None,
        adaptation_steps=1,
    ):
        if max_pool_factor is None:
            max_pool_factor = 4 // layers
        super(CNN4Backbone_BNRS, self).__init__(
            hidden=hidden_size,
            layers=layers,
            channels=channels,
            max_pool=max_pool,
            max_pool_factor=max_pool_factor,
            adaptation_steps=adaptation_steps
        )

    def forward(self, x, step):
        x = super(CNN4Backbone_BNRS, self).forward(x, step)
        x = x.reshape(x.size(0), -1)
        return x


class CNN4_BNRS(torch.nn.Module):
    """

    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/vision/models/cnn4.py)

    **Description**

    The convolutional network commonly used for MiniImagenet, as described by Ravi et Larochelle, 2017.

    This network assumes inputs of shapes (3, 84, 84).

    Instantiate `CNN4Backbone` if you only need the feature extractor.

    **References**

    1. Ravi and Larochelle. 2017. “Optimization as a Model for Few-Shot Learning.” ICLR.

    **Arguments**

    * **output_size** (int) - The dimensionality of the network's output.
    * **hidden_size** (int, *optional*, default=64) - The dimensionality of the hidden representation.
    * **layers** (int, *optional*, default=4) - The number of convolutional layers.
    * **channels** (int, *optional*, default=3) - The number of channels in input.
    * **max_pool** (bool, *optional*, default=True) - Whether ConvBlocks use max-pooling.
    * **embedding_size** (int, *optional*, default=None) - Size of feature embedding.
        Defaults to 25 * hidden_size (for mini-Imagenet).

    **Example**
    ~~~python
    model = CNN4(output_size=20, hidden_size=128, layers=3)
    ~~~
    """

    def __init__(
        self,
        output_size,
        hidden_size=64,
        layers=4,
        channels=3,
        max_pool=True,
        embedding_size=None,
        adaptation_steps=1,
    ):
        super(CNN4_BNRS, self).__init__()
        if embedding_size is None:
            embedding_size = 25 * hidden_size
        self.features = CNN4Backbone_BNRS(
            hidden_size=hidden_size,
            channels=channels,
            max_pool=max_pool,
            layers=layers,
            max_pool_factor=4 // layers,
            adaptation_steps=adaptation_steps,
        )
        self.classifier = torch.nn.Linear(
            embedding_size,
            output_size,
            bias=True,
        )
        maml_init_(self.classifier)
        self.hidden_size = hidden_size

    def backup_stats(self):
        """
        Backup stored batch statistics before running a validation epoch.
        """
        for layer in self.features.modules():
            if type(layer) is MetaBatchNormLayer:
                layer.backup_stats()

    def restore_backup_stats(self):
        """
        Reset stored batch statistics from the stored backup.
        """
        for layer in self.features.modules():
            if type(layer) is MetaBatchNormLayer:
                layer.restore_backup_stats()

    def forward(self, x, step):
        x = self.features(x, step)
        x = self.classifier(x)
        return x
