#!/usr/bin/env python3

import torch as th
from torch import nn
from torch.autograd import grad

from learn2learn.utils import clone_module, clone_parameters
from learn2learn.algorithms.base_learner import BaseLearner


def meta_sgd_update(model, lrs=None, grads=None):
    if grads is not None and lrs is not None:
        for p, lr, g in zip(model.parameters(), lrs, grads):
            p.grad = g
            p._lr = lr

    # Update the params
    for param_key in model._parameters:
        p = model._parameters[param_key]
        if p is not None and p.grad is not None:
            model._parameters[param_key] = p - p._lr * p.grad

    # Second, handle the buffers if necessary
    for buffer_key in model._buffers:
        buff = model._buffers[buffer_key]
        if buff is not None and buff.grad is not None and buff._lr is not None:
            model._buffers[buffer_key] = buff - buff._lr * buff.grad

    # Then, recurse for each submodule
    for module_key in model._modules:
        model._modules[module_key] = meta_sgd_update(model._modules[module_key])
    return model


class MetaSGD(BaseLearner):

    def __init__(self, model, lr=1.0, first_order=False, lrs=None):
        super(MetaSGD, self).__init__()
        self.module = model
        if lrs is None:
            lrs = [th.ones_like(p) * lr for p in model.parameters()]
            lrs = nn.ParameterList([nn.Parameter(lr) for lr in lrs])
        self.lrs = lrs
        self.first_order = first_order

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def clone(self):
        return MetaSGD(clone_module(self.module),
                       lrs=clone_parameters(self.lrs),
                       first_order=self.first_order)

    def adapt(self, loss, first_order=None):
        if first_order is None:
            first_order = self.first_order
        second_order = not first_order
        gradients = grad(loss,
                         self.module.parameters(),
                         retain_graph=second_order,
                         create_graph=second_order)
        self.module = meta_sgd_update(self.module, self.lrs, gradients)


if __name__ == '__main__':
    linear = nn.Sequential(nn.Linear(10, 2), nn.Linear(5, 5))
    msgd = MetaSGD(linear, lr=0.001)
    learner = msgd.new()
