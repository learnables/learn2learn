#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#

"""
BatchNorm layer augmented with Per-Step Batch Normalisation Running Statistics and Per-Step Batch
Normalisation Weights and Biases, as proposed in MAML++ by Antobiou et al.
"""

import torch
import torch.nn.functional as F

from copy import deepcopy


class MetaBatchNorm(torch.nn.Module):
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/nn/metabatchnorm.py)

    **Description**

    An extension of Pytorch's BatchNorm layer, with the Per-Step Batch Normalisation Running
    Statistics and Per-Step Batch Normalisation Weights and Biases improvements proposed in
    "How to train your MAML".
    It is adapted from the original Pytorch implementation at
    https://github.com/AntreasAntoniou/HowToTrainYourMAMLPytorch,
    with heavy refactoring and a bug fix
    (https://github.com/AntreasAntoniou/HowToTrainYourMAMLPytorch/issues/42).

    **Arguments**

    * **num_features** (int) - number of input features.
    * **adaptation_steps** (int) - number of inner-loop adaptation steps.
    * **eps** (float, *optional*, default=1e-5) - a value added to the denominator for numerical
    stability.
    * **momentum** (float, *optional*, default=0.1) - the value used for the running_mean and
    running_var computation. Can be set to None for cumulative moving average (i.e. simple
    average).
    * **affine** (bool, *optional*, default=True) - a boolean value that when set to True, this
    module has learnable affine parameters.

    **References**

    1. Antoniou et al. 2019. "How to train your MAML." ICLR.

    **Example**

    ~~~python
    batch_norm = MetaBatchNorm(100, 5)
    input = torch.randn(20, 100, 35, 45)
    for step in range(5):
        output = batch_norm(input, step)
    ~~~
    """

    def __init__(
        self,
        num_features,
        adaptation_steps,
        eps=1e-5,
        momentum=0.1,
        affine=True,
    ):
        super(MetaBatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
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
        self._steps = adaptation_steps
        self._current_step = 0

    def forward(
        self,
        input,
        inference=False,
    ):
        """
        **Arguments**

        * **input** (tensor) - Input data batch, size either can be any.
        * **inferencep** (bool, *optional*, default=False) - when set to `True`, uses the final
        step's parameters and running statistics. When set to `False`, automatically infers the
        current adaptation step.
        """
        step = self._current_step if not inference else self._steps - 1
        output = F.batch_norm(
            input,
            self.running_mean[step],
            self.running_var[step],
            self.weight[step],
            self.bias[step],
            training=True,
            momentum=self.momentum,
            eps=self.eps,
        )
        if not inference:
            self._current_step = (
                self._current_step + 1 if self._current_step < (self._steps - 1) else 0
            )
        return output

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
