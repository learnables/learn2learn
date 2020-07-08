#!/usr/bin/env python3

import torch
from learn2learn.utils import update_module


class DifferentiableSGD(torch.nn.Module):
    r"""
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/optim/update_rules/differentiable_sgd.py)

    **Description**

    A callable object that applies a list of updates to the parameters of a torch.nn.Module in a differentiable manner.

    For each parameter \(p\) and corresponding gradient \(g\), calling an instance of this class results in updating parameters:

    \[
    p \gets p - \alpha g,
    \]

    where \(\alpha\) is the learning rate.

    Note: The module is updated in-place.

    **Arguments**

    * **lr** (float) - The learning rate used to update the model.

    **Example**
    ~~~python
    sgd = DifferentiableSGD(0.1)
    gradients = torch.autograd.grad(
        loss,
        model.parameters(),
        create_gaph=True)
    sgd(model, gradients)  # model is updated in-place
    ~~~
    """

    def __init__(self, lr):
        super(DifferentiableSGD, self).__init__()
        self.lr = lr

    def forward(self, module, gradients=None):
        """
        **Arguments**

        * **module** (Module) - The module to update.
        * **gradients** (list, *optional*, default=None) - A list of gradients for each parameter
            of the module. If None, will use the gradients in .grad attributes.

        """
        if gradients is None:
            gradients = [p.grad for p in module.parameters()]
        updates = [None if g is None else g.mul(-self.lr)
                   for g in gradients]
        update_module(module, updates)
