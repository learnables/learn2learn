#!/usr/bin/env python3

import torch as th
from torch import nn
from torch.autograd import grad

from learn2learn.maml import clone_module


def clone_parameters(param_list):
    return [p.clone() for p in param_list]


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


class MetaSGDLearner(nn.Module):

    def __init__(self, module, lrs, first_order):
        super(MetaSGDLearner, self).__init__()
        self.module = module
        self.lrs = lrs
        self.first_order = first_order

    def __getattr__(self, attr):
        try:
            return super(MetaSGDLearner, self).__getattr__(attr)
        except AttributeError:
            return getattr(self.__dict__['_modules']['module'], attr)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def adapt(self, loss, first_order=None):
        if first_order is None:
            first_order = self.first_order
        second_order = not first_order
        gradients = grad(loss,
                         self.module.parameters(),
                         retain_graph=second_order,
                         create_graph=second_order)
        self.module = meta_sgd_update(self.module, self.lrs, gradients)


class MetaSGD(nn.Module):

    def __init__(self, model, lr=1.0, first_order=False):
        super(MetaSGD, self).__init__()
        self.model = model
        self.lrs = [th.ones_like(p) * lr for p in model.parameters()]
        self.lrs = nn.ParameterList([nn.Parameter(lr) for lr in self.lrs])
        self.first_order = first_order

    def forward(self, *args, **kwargs):
        return self.new(*args, **kwargs)

    def new(self):
        return MetaSGDLearner(module=clone_module(self.model),
                              lrs=clone_parameters(self.lrs),
                              first_order=self.first_order)


if __name__ == '__main__':
    linear = nn.Sequential(nn.Linear(10, 2), nn.Linear(5, 5))
    msgd = MetaSGD(linear, lr=0.001)
    learner = msgd.new()
