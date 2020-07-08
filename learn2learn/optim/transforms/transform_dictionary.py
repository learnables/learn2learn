#!/usr/bin/env python3

import torch


class TransformDictionary(object):
    """docstring for ModuleDependentTransform"""

    def __init__(self, dictionary):
        self.param_to_transform = {}
        for key, transform in dictionary.items():
            if isinstance(key, torch.nn.Module):
                for p in key.parameters():
                    self.param_to_transform[p] = transform
            elif isinstance(key, torch.nn.Parameter):
                self.param_to_transform[key] = transform
            else:
                raise ValueError(
                        'TransformDictionary only accepts Modules' +
                        ' or Parameters as dictionary keys.')

    def __call__(self, param, *args, **kwargs):
        if param in self.param_to_transform:
            return self.param_to_transform[param](param, *args, **kwargs)
        return None
