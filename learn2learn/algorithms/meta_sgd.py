#!/usr/bin/env python3

import torch as th
from torch import nn
from torch.autograd import grad

from learn2learn.algorithms.base_learner import BaseLearner
from learn2learn.utils import clone_module, clone_parameters


def meta_sgd_update(model, lrs=None, grads=None):
    """

    **Description**

    Performs a MetaSGD update on model using grads and lrs.
    The function re-routes the Python object, thus avoiding in-place
    operations.

    NOTE: The model itself is updated in-place (no deepcopy), but the
          parameters' tensors are not.

    **Arguments**

    * **model** (Module) - The model to update.
    * **lrs** (list) - The meta-learned learning rates used to update the model.
    * **grads** (list, *optional*, default=None) - A list of gradients for each parameter
        of the model. If None, will use the gradients in .grad attributes.

    **Example**
    ~~~python
    meta = l2l.algorithms.MetaSGD(Model(), lr=1.0)
    lrs = [th.ones_like(p) for p in meta.model.parameters()]
    model = meta.clone() # The next two lines essentially implement model.adapt(loss)
    grads = autograd.grad(loss, model.parameters(), create_graph=True)
    meta_sgd_update(model, lrs=lrs, grads)
    ~~~
    """
    if grads is not None and lrs is not None:
        for p, lr, g in zip(model.parameters(), lrs, grads):
            p.grad = g
            p._lr = lr

    # Update the params
    for param_key in model._parameters:
        p = model._parameters[param_key]
        if p is not None and p.grad is not None:
            model._parameters[param_key] = p - p._lr * p.grad
            p.grad = None
            p._lr = None

    # Second, handle the buffers if necessary
    for buffer_key in model._buffers:
        buff = model._buffers[buffer_key]
        if buff is not None and buff.grad is not None and buff._lr is not None:
            model._buffers[buffer_key] = buff - buff._lr * buff.grad
            buff.grad = None
            buff._lr = None

    # Then, recurse for each submodule
    for module_key in model._modules:
        model._modules[module_key] = meta_sgd_update(model._modules[module_key])
    return model


class MetaSGD(BaseLearner):
    """

    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/algorithms/meta_sgd.py)

    **Description**

    High-level implementation of *Meta-SGD*.

    This class wraps an arbitrary nn.Module and augments it with `clone()` and `adapt`
    methods.
    It behaves similarly to `MAML`, but in addition a set of per-parameters learning rates
    are learned for fast-adaptation.

    **Arguments**

    * **model** (Module) - Module to be wrapped.
    * **lr** (float) - Initialization value of the per-parameter fast adaptation learning rates.
    * **first_order** (bool, *optional*, default=False) - Whether to use the first-order version.
    * **lrs** (list of Parameters, *optional*, default=None) - If not None, overrides `lr`, and uses the list
        as learning rates for fast-adaptation.

    **References**

    1. Li et al. 2017. “Meta-SGD: Learning to Learn Quickly for Few-Shot Learning.” arXiv.

    **Example**

    ~~~python
    linear = l2l.algorithms.MetaSGD(nn.Linear(20, 10), lr=0.01)
    clone = linear.clone()
    error = loss(clone(X), y)
    clone.adapt(error)
    error = loss(clone(X), y)
    error.backward()
    ~~~
    """

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
        """
        **Descritpion**

        Akin to `MAML.clone()` but for MetaSGD: it includes a set of learnable fast-adaptation
        learning rates.
        """
        return MetaSGD(clone_module(self.module),
                       lrs=clone_parameters(self.lrs),
                       first_order=self.first_order)

    def adapt(self, loss, first_order=None):
        """
        **Descritpion**

        Akin to `MAML.adapt()` but for MetaSGD: it updates the model with the learnable
        per-parameter learning rates.
        """
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
