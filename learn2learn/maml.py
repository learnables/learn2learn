#!/usr/bin/env python3

import copy

from torch import nn
from torch.autograd import grad


def clone_module(module):
    """
    Creates a copy of a module, whose parameters/buffers/submodules
    are created using PyTorch's th.clone().

    This implies that the computational graph is kept, and you can compute
    the derivatives of the new modules' parameters w.r.t the original
    parameters.

    NOTE: This function might break in future versions of PyTorch.

    TODO: This function might require that module.forward()
          was called in order to work properly, if forward() instanciates
          new variables.
    TODO: deepcopy is expensive. We can probably get away with a shallowcopy.
          However, since shallow copy does not recurse, we need to write a
          recursive version of shallow copy.
    NOTE: This can probably be implemented more cleanly with
          clone = recursive_shallow_copy(model)
          clone._apply(lambda t: t.clone())
    """
    clone = copy.deepcopy(module)

    # First, re-write all parameters
    for param_key in module._parameters:
        if module._parameters[param_key] is not None:
            cloned = module._parameters[param_key].clone()
            clone._parameters[param_key] = cloned

    # Second, handle the buffers if necessary
    for buffer_key in module._buffers:
        if clone._buffers[buffer_key] is not None and \
                clone._buffers[buffer_key].requires_grad:
            clone._buffers[buffer_key] = module._buffers[buffer_key].clone()

    # Then, recurse for each submodule
    for module_key in clone._modules:
        clone._modules[module_key] = clone_module(module._modules[module_key])
    return clone


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


class MAMLLearner(nn.Module):
    def __init__(self, module, lr, first_order=False):
        super(MAMLLearner, self).__init__()
        self.module = module
        self.lr = lr
        self.second_order = not first_order

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def adapt(self, loss):
        gradients = grad(loss,
                         self.module.parameters(),
                         retain_graph=self.second_order,
                         create_graph=self.second_order)
        self.module = maml_update(self.module, self.lr, gradients)


class MAML(nn.Module):

    def __init__(self, model, lr, first_order=False):
        super(MAML, self).__init__()
        self.model = model
        self.lr = lr
        self.first_order = first_order

    def forward(self, *args, **kwargs):
        return self.new(*args, **kwargs)

    def new(self, first_order=None):
        if first_order is None:
            first_order = self.first_order
        return MAMLLearner(clone_module(self.model),
                           lr=self.lr,
                           first_order=first_order)
