#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#

"""
Learning Per-Layer Per-Step Learning Rates (LSLR, MAML++).
"""

import torch


def update_with_lslr(model: torch.nn.Module, lrs=None, grads=None, step=None):
    """

    **Description**

    Performs an update on model using grads and LSLR.
    The function re-routes the Python object, thus avoiding in-place
    operations.

    NOTE: The model itself is updated in-place (no deepcopy), but the
          parameters' tensors are not.

    **Arguments**

    * **model** (Module) - The model to update.
    * **lrs** (list) - The meta-learned per-layer per-step learning rates used to update the model.
    * **grads** (list, *optional*, default=None) - A list of gradients for each layer
        of the model. If None, will use the gradients in .grad attributes.
    * **step** (int) - The current adaptation step, used for the LSLR.
    """
    if grads is not None and lrs is not None:
        params = list(model.parameters())
        if not len(grads) == len(list(params)):
            msg = "WARNING:maml_update(): Parameters and gradients have different length. ("
            msg += str(len(params)) + " vs " + str(len(grads)) + ")"
            print(msg)
        # TODO: Why doesn't this work?? I can't assign p.grad when zipping like this... Is this
        # because I'm using a tuple?
        # for named_param, g in zip(
            # [(k, v) for k, l in model.named_parameters() for v in l], grads
        # ):
            # p_name, p = named_param
        it = 0
        for name, p in model.named_parameters():
            if grads[it] is not None:
                lr = None
                layer_name = name[: name.rfind(".")].replace(
                    ".", "-"
                )  # Extract the layer name from the named parameter
                lr = lrs[layer_name][step]
                assert (
                    lr is not None
                ), f"Parameter {name} does not have a learning rate in LSLR dict!"
                p.grad = grads[it]
                p._lr = lr
            it += 1

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
        model._modules[module_key] = update_with_lslr(model._modules[module_key])
    return model



class LSLR(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, adaptation_steps: int, init_lr: float):
        super(LSLR, self).__init__()
        module.update_func = update_with_lslr
        self.model = module
        self.lrs = self._init_lslr_parameters(
            adaptation_steps=adaptation_steps, init_lr=init_lr
        )

    def _init_lslr_parameters(
        self, adaptation_steps: int, init_lr: float
    ) -> torch.nn.ParameterDict:
        lslr = torch.nn.ParameterDict()
        for layer_name, layer in self.model.named_modules():
            # If the layer has learnable parameters
            if (
                len(
                    [
                        name
                        for name, param in layer.named_parameters(recurse=False)
                        if param.requires_grad
                    ]
                )
                > 0
            ):
                lslr[layer_name.replace(".", "-")] = torch.nn.Parameter(
                    data=torch.ones(adaptation_steps) * init_lr,
                    requires_grad=True,
                )
        return lslr

    def __getattr__(self, name):
        if name == "model":
            # TODO: Find a proper way to do this hacky thing?
            attr = self.__dict__['_modules']['model']
            # attr = super(LSLR, self).__getattribute__(name)
            # attr = object.__getattribute__(self, name)
        else:
            attr = object.__getattribute__(self.model, name)
        return attr
