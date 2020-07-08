#!/usr/bin/env python3

import torch


class ReshapedTransform(torch.nn.Module):
    """
    Helper class to reshape gradients before they are fed to a Module and
    reshape back the update returned by the Module.
    """

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

    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/optim/transforms/module_transform.py)

    **Description**

    The ModuleTransform creates a an optimization transform based on any nn.Module.

    ModuleTransform automatically instanciates a module from its class, based on a given parameter.
    The input and output shapes are of the module are set to `(1, param.numel())`.

    When optimizing large layers, this type of transform can quickly run out of memory.
    See `KroneckerTransform` for a scalable alternative.

    **Arguments**

    * **module_cls** (callable) - A callable that instantiates the module used to transform gradients.

    **Example**
    ~~~python
    classifier = torch.nn.Linear(784, 10, bias=False)
    linear_transform = ModuleTransform(torch.nn.Linear)
    linear_update = linear_transform(classifier.weight)  # maps gradients to updates, both of shape (1, 7840)
    loss(classifier(X), y).backward()
    update = linear_update(classifier.weight.grad)
    classifier.weight.data.add_(-lr, update)  # Not a differentiable update. See l2l.optim.DifferentiableSGD.
    ~~~
    """

    def __init__(self, module_cls):
        self.module_cls = module_cls

    def __call__(self, parameter):
        numel = parameter.numel()
        flat_shape = (1, numel)
        transform = self.module_cls(numel, numel)
        return ReshapedTransform(
            transform=transform,
            shape=flat_shape,
        )
