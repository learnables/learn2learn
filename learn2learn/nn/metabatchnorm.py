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
        adaptation_steps,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        meta_batch_norm=True,
    ):
        super(MetaBatchNorm, self).__init__()
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
