#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#

"""
Learning Per-Layer Per-Step Learning Rates (LSLR, MAML++).
"""

import torch


from learn2learn.utils import clone_named_parameters, update_module


class LSLR:
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/optim/lslr.py)

    **Description**

    Implementation of *Learning Per-Layer Per-Step Learning Rates* from MAML++ as a l2l algorithm
    wrapper.

    This class wraps an arbitrary l2l algorithm implementing an adapt method and overrides its
    clone() and adapt() calls.
    It computes parameters updates, but in addition a set of per-parameters learning rates
    are learned for fast-adaptation.

    **Arguments**

    * **model** (Module) - Module (l2l algorithm) to be wrapped.
    * **adaptation_steps** (int) - Number of inner-loop adaptation steps.
    * **init_lr** (float) - Initial learning rate for all layers and steps.

    **Example**

    ~~~python
    linear = l2l.optim.LSLR(
        l2l.algorithms.MetaSGD(nn.Linear(20, 10), lr=0.01), # The algorithm to wrap
        adaptation_steps=1,
        init_lr=0.01
    )
    clone = linear.clone()
    error = loss(clone(X), y)
    clone.adapt(error)
    error = loss(clone(X), y)
    error.backward()
    ~~~
    """

    def __init__(self, module: torch.nn.Module, adaptation_steps: int, init_lr: float):
        module.update_func = self.update_with_lslr
        self.model = module
        self.model.lslr = self._init_lslr_parameters(
            adaptation_steps=adaptation_steps, init_lr=init_lr
        )
        self._current_step = 0

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
                lslr[layer_name.replace("module.", "").replace(".", "-")] = torch.nn.Parameter(
                    data=torch.ones(adaptation_steps) * init_lr,
                    requires_grad=True,
                )
        return lslr

    def update_with_lslr(self, model: torch.nn.Module, lr=None, grads=None, **kwargs):
        # TODO: Turn this into a GBML gradient transform instead?
        """

        **Description**

        Performs an update on model using grads and LSLR.
        The function re-routes the Python object, thus avoiding in-place
        operations.

        NOTE: The model itself is updated in-place (no deepcopy), but the
              parameters' tensors are not.

        **Arguments**

        * **model** (Module) - The model to update.
        * **grads** (list, *optional*, default=None) - A list of gradients for each layer
            of the model. If None, will use the gradients in .grad attributes.
        """
        if grads is not None:
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
                    lr = self.model.lslr[layer_name][self._current_step]
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
            model._modules[module_key] = self.update_with_lslr(model._modules[module_key])
        return model

    def __getattr__(self, name):
        if name == "clone":
            def override(*args, **kwargs):
                method = object.__getattribute__(self.model, name)
                lslr = {k: p.clone() for k, p in self.model.lslr.items()}
                # Original clone method
                self.model = method(*args, **kwargs)
                # Override the update function to LSLR
                with torch.no_grad():
                    self.model.update_func = self.update_with_lslr
                    self.model.lslr = lslr
                return self
            attr = override
        elif name == "adapt":
            def override(*args, **kwargs):
                assert 'step' in kwargs, "Keyword argument 'step' not passed to the adapt() method"
                with torch.no_grad():
                    self._current_step = kwargs['step']
                    del kwargs['step']
                method = object.__getattribute__(self.model, name)
                method(*args, **kwargs)
            attr = override
        else:
            with torch.no_grad():
                attr = object.__getattribute__(self.model, name)
        return attr


    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)



