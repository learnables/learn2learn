#!/usr/bin/env python3

import torch


def magic_box(x):
    if isinstance(x, torch.Tensor):
        return torch.exp(x - x.detach())
    return x
