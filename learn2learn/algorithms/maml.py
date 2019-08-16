#!/usr/bin/env python3

from torch.autograd import grad

from learn2learn import clone_module
from learn2learn.algorithms.base_learner import BaseLearner


def maml_update(model, lr, grads):
    """
    Performs a MAML update on model using grads and lr.
    The function re-routes the Python object, thus avoiding in-place
    operations. However, it seems like PyTorch handles in-place operations
    fairly well.

    NOTE: grads is None -> Don't set the gradients.
    """
    if grads is not None:
        for p, g in zip(model.parameters(), grads):
            p.grad = g

    # Update the params
    for param_key in model._parameters:
        p = model._parameters[param_key]
        if p is not None and p.grad is not None:
            model._parameters[param_key] = p - lr * p.grad

    # Second, handle the buffers if necessary
    for buffer_key in model._buffers:
        buff = model._buffers[buffer_key]
        if buff is not None and buff.grad is not None:
            model._buffers[buffer_key] = buff - lr * buff.grad

    # Then, recurse for each submodule
    for module_key in model._modules:
        model._modules[module_key] = maml_update(model._modules[module_key],
                                                 lr=lr,
                                                 grads=None)
    return model


class MAML(BaseLearner):

    def __init__(self, model, lr, first_order=False):
        super(MAML, self).__init__()
        self.module = model
        self.lr = lr
        self.first_order = first_order

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
        self.module = maml_update(self.module, self.lr, gradients)

    def clone(self, first_order=None):
        if first_order is None:
            first_order = self.first_order
        return MAML(clone_module(self.module),
                    lr=self.lr,
                    first_order=first_order)
