#!/usr/bin/env python3

import torch


class ReshapedTransform(torch.nn.Module):
    """docstring for ReshaperTransform"""

    def __init__(self, transform, shape):
        super(ReshapedTransform, self).__init__()
        self.transform = transform
        self.in_shape = shape

    def forward(self, grad):
        """docstring for __forward__"""
        out_shape = grad.shape
        update = grad.view(self.in_shape)
        update = self.transform(update)
        update = update.view(out_shape)
        return update


class ModuleTransform(object):
    """docstring for ModuleTransform"""

    def __init__(self, module_cls):
        self.module_cls = module_cls

    def __call__(self, parameter):
        """docstring for __call__"""
        numel = parameter.numel()
        flat_shape = (1, numel)
        transform = self.module_cls(numel, numel)
        return ReshapedTransform(
            transform=transform,
            shape=flat_shape,
        )
