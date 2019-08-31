#!/usr/bin/env python3

import copy


def clone_parameters(param_list):
    return [p.clone() for p in param_list]


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


def detach_module(module):
    severed = copy.deepcopy(module)

    # First, re-write all parameters
    for param_key in module._parameters:
        if module._parameters[param_key] is not None:
            detached = module._parameters[param_key].detach()
            severed._parameters[param_key] = detached

    # Second, handle the buffers if necessary
    for buffer_key in module._buffers:
        if severed._buffers[buffer_key] is not None and \
                severed._buffers[buffer_key].requires_grad:
            severed._buffers[buffer_key] = module._buffers[buffer_key].detach()

    # Then, recurse for each submodule
    for module_key in severed._modules:
        severed._modules[module_key] = detach_module(module._modules[module_key])
    return severed


def clone_distribution(dist):
    clone = copy.deepcopy(dist)

    for param_key in clone.__dict__:
        item = clone.__dict__[param_key]
        if isinstance(item, th.Tensor):
            if item.requires_grad:
                clone.__dict__[param_key] = dist.__dict__[param_key].clone()
        elif isinstance(item, th.nn.Module):
            clone.__dict__[param_key] = clone_module(dist.__dict__[param_key])
        elif isinstance(item, th.Distribution):
            clone.__dict__[param_key] = clone_distribution(dist.__dict__[param_key])

    return clone


def detach_distribution(dist):

    severed = copy.deepcopy(dist)

    for param_key in severed.__dict__:
        item = severed.__dict__[param_key]
        if isinstance(item, th.Tensor):
            if item.requires_grad:
                severed.__dict__[param_key] = dist.__dict__[param_key].detach()
        elif isinstance(item, th.nn.Module):
            severed.__dict__[param_key] = detach_module(dist.__dict__[param_key])
        elif isinstance(item, th.Distribution):
            severed.__dict__[param_key] = detach_distribution(dist.__dict__[param_key])

    return severed
