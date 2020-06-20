#!/usr/bin/env python3

import torch
from learn2learn.utils import meta_update


class DifferentiableSGD(torch.nn.Module):
    """docstring for DifferentiableSGD"""

    def __init__(self, lr):
        super(DifferentiableSGD, self).__init__()
        self.lr = lr

    def forward(self, module, gradients=None):
        """docstring for forward"""
        if gradients is None:
            gradients = [p.grad for p in module.parameters()]
        updates = [None if g is None else g.mul(-self.lr)
                   for g in gradients]
        meta_update(module, updates)
