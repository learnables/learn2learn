#!/usr/bin/env python3

import torch


class Lambda(torch.nn.Module):
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/nn/misc.py)

    **Description**

    Utility class to create a wrapper based on a lambda function.

    **Arguments**

    * **lmb** (callable) - The function to call in the forward pass.

    **Example**
    ~~~python
    mean23 = Lambda(lambda x: x.mean(dim=[2, 3]))  # mean23 is a Module
    x = features(img)
    x = mean23(x)
    x = x.flatten()
    ~~~
    """

    def __init__(self, lmb):
        super(Lambda, self).__init__()
        self.lmb = lmb

    def forward(self, *args, **kwargs):
        return self.lmb(*args, **kwargs)


class Flatten(torch.nn.Module):
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/nn/misc.py)

    **Description**

    Utility Module to flatten inputs to `(batch_size, -1)` shape.

    **Example**
    ~~~python
    flatten = Flatten()
    x = torch.randn(5, 3, 32, 32)
    x = flatten(x)
    print(x.shape)  # (5, 3072)
    ~~~
    """

    def forward(self, x):
        return x.view(x.size(0), -1)


class Scale(torch.nn.Module):
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/nn/misc.py)

    **Description**

    A per-parameter scaling factor with learnable parameter.

    **Arguments**

    * **shape** (int or tuple, *optional*, default=1) - The shape of the scaling matrix.
    * **alpha** (float, *optional*, default=1.0) - Initial value for the
        scaling factor.

    **Example**
    ~~~python
    x = torch.ones(3)
    scale = Scale(x.shape, alpha=0.5)
    print(scale(x))  # [.5, .5, .5]
    ~~~
    """

    def __init__(self, shape=None, alpha=1.0):
        super(Scale, self).__init__()
        if isinstance(shape, int):
            shape = (shape, )
        elif shape is None:
            shape = (1, )
        alpha = torch.ones(*shape) * alpha
        self.alpha = torch.nn.Parameter(alpha)

    def forward(self, x):
        return x * self.alpha


def freeze(module):
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/nn/misc.py)

    **Description**

    Prevents all parameters in `module` to get gradients.

    Note: the module is modified in-place.

    **Arguments**

    * **module** (Module) - The module to freeze.

    **Example**
    ~~~python
    linear = torch.nn.Linear(128, 4)
    l2l.nn.freeze(linear)
    ~~~
    """
    for p in module.parameters():
        p.detach_()
        if hasattr(p, 'requires_grad'):
            p.requires_grad = False
    return module


def unfreeze(module):
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/nn/misc.py)

    **Description**

    Enables all parameters in `module` to compute gradients.

    Note: the module is modified in-place.

    **Arguments**

    * **module** (Module) - The module to unfreeze.

    **Example**

    ~~~python
    linear = torch.nn.Linear(128, 4)
    l2l.nn.freeze(linear)
    l2l.nn.unfreeze(linear)
    ~~~
    """
    for p in module.parameters():
        if hasattr(p, 'requires_grad'):
            p.requires_grad = True
    return module
