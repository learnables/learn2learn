#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#

"""
MAML++ wrapper.
"""

import torch
import traceback

from torch.autograd import grad

from learn2learn.algorithms.base_learner import BaseLearner
from learn2learn.utils import clone_module, update_module, clone_named_parameters


def maml_pp_update(model, step=None, lrs=None, grads=None):
    """

    **Description**

    Performs a MAML++ update on model using grads and lrs.
    The function re-routes the Python object, thus avoiding in-place
    operations.

    NOTE: The model itself is updated in-place (no deepcopy), but the
          parameters' tensors are not.

    **Arguments**

    * **model** (Module) - The model to update.
    * **lrs** (list) - The meta-learned learning rates used to update the model.
    * **grads** (list, *optional*, default=None) - A list of gradients for each layer
        of the model. If None, will use the gradients in .grad attributes.

    **Example**
    ~~~python
    maml_pp = l2l.algorithms.MAMLpp(Model(), lr=1.0)
    lslr = torch.nn.ParameterDict()
    for layer_name, layer in model.named_modules():
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
    model = maml_pp.clone() # The next two lines essentially implement model.adapt(loss)
    for inner_step in range(5):
        loss = criterion(model(x), y)
        grads = autograd.grad(loss, model.parameters(), create_graph=True)
        maml_pp_update(model, inner_step, lrs=lslr, grads=grads)
    ~~~
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
        model._modules[module_key] = maml_pp_update(model._modules[module_key])
    return model


class MAMLpp(BaseLearner):
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/algorithms/maml.py)

    **Description**

    High-level implementation of *Model-Agnostic Meta-Learning*.

    This class wraps an arbitrary nn.Module and augments it with `clone()` and `adapt()`
    methods.

    For the first-order version of MAML (i.e. FOMAML), set the `first_order` flag to `True`
    upon initialization.

    **Arguments**

    * **model** (Module) - Module to be wrapped.
    * **lr** (float) - Fast adaptation learning rate.
    * **lslr** (bool) - Whether to use Per-Layer Per-Step Learning Rates and Gradient Directions
        (LSLR) or not.
    * **lrs** (list of Parameters, *optional*, default=None) - If not None, overrides `lr`, and uses the list
        as learning rates for fast-adaptation.
    * **first_order** (bool, *optional*, default=False) - Whether to use the first-order
        approximation of MAML. (FOMAML)
    * **allow_unused** (bool, *optional*, default=None) - Whether to allow differentiation
        of unused parameters. Defaults to `allow_nograd`.
    * **allow_nograd** (bool, *optional*, default=False) - Whether to allow adaptation with
        parameters that have `requires_grad = False`.

    **References**

    1. Finn et al. 2017. "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks."

    **Example**

    ~~~python
    linear = l2l.algorithms.MAML(nn.Linear(20, 10), lr=0.01)
    clone = linear.clone()
    error = loss(clone(X), y)
    clone.adapt(error)
    error = loss(clone(X), y)
    error.backward()
    ~~~
    """

    def __init__(
        self,
        model,
        lr,
        lrs=None,
        adaptation_steps=1,
        first_order=False,
        allow_unused=None,
        allow_nograd=False,
    ):
        super().__init__()
        self.module = model
        self.lr = lr
        if lrs is None:
            lrs = self._init_lslr_parameters(model, adaptation_steps, lr)
        self.lrs = lrs
        self.first_order = first_order
        self.allow_nograd = allow_nograd
        if allow_unused is None:
            allow_unused = allow_nograd
        self.allow_unused = allow_unused

    def _init_lslr_parameters(
        self, model: torch.nn.Module, adaptation_steps: int, init_lr: float
    ) -> torch.nn.ParameterDict:
        lslr = torch.nn.ParameterDict()
        for layer_name, layer in model.named_modules():
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

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def adapt(self, loss, step=None, first_order=None, allow_unused=None, allow_nograd=None):
        """
        **Description**

        Takes a gradient step on the loss and updates the cloned parameters in place.

        **Arguments**

        * **loss** (Tensor) - Loss to minimize upon update.
        * **step** (int) - Current inner loop step. Used to fetch the corresponding learning rate.
        * **first_order** (bool, *optional*, default=None) - Whether to use first- or
            second-order updates. Defaults to self.first_order.
        * **allow_unused** (bool, *optional*, default=None) - Whether to allow differentiation
            of unused parameters. Defaults to self.allow_unused.
        * **allow_nograd** (bool, *optional*, default=None) - Whether to allow adaptation with
            parameters that have `requires_grad = False`. Defaults to self.allow_nograd.
        """
        if first_order is None:
            first_order = self.first_order
        if allow_unused is None:
            allow_unused = self.allow_unused
        if allow_nograd is None:
            allow_nograd = self.allow_nograd
        second_order = not first_order

        gradients = []
        if allow_nograd:
            # Compute relevant gradients
            diff_params = [p for p in self.module.parameters() if p.requires_grad]
            grad_params = grad(
                loss,
                diff_params,
                retain_graph=second_order,
                create_graph=second_order,
                allow_unused=allow_unused,
            )
            grad_counter = 0

            # Handles gradients for non-differentiable parameters
            for param in self.module.parameters():
                if param.requires_grad:
                    gradient = grad_params[grad_counter]
                    grad_counter += 1
                else:
                    gradient = None
                gradients.append(gradient)
        else:
            try:
                gradients = grad(
                    loss,
                    self.module.parameters(),
                    retain_graph=second_order,
                    create_graph=second_order,
                    allow_unused=allow_unused,
                )
            except RuntimeError:
                traceback.print_exc()
                print(
                    "learn2learn: Maybe try with allow_nograd=True and/or allow_unused=True ?"
                )

        # Update the module
        assert step is not None, "step cannot be None when using LSLR!"
        self.module = maml_pp_update(self.module, step, lrs=self.lrs, grads=gradients)

    def clone(self, first_order=None, allow_unused=None, allow_nograd=None):
        """
        **Description**

        Returns a `MAMLpp`-wrapped copy of the module whose parameters and buffers
        are `torch.clone`d from the original module.

        This implies that back-propagating losses on the cloned module will
        populate the buffers of the original module.
        For more information, refer to learn2learn.clone_module().

        **Arguments**

        * **first_order** (bool, *optional*, default=None) - Whether the clone uses first-
            or second-order updates. Defaults to self.first_order.
        * **allow_unused** (bool, *optional*, default=None) - Whether to allow differentiation
        of unused parameters. Defaults to self.allow_unused.
        * **allow_nograd** (bool, *optional*, default=False) - Whether to allow adaptation with
            parameters that have `requires_grad = False`. Defaults to self.allow_nograd.

        """
        if first_order is None:
            first_order = self.first_order
        if allow_unused is None:
            allow_unused = self.allow_unused
        if allow_nograd is None:
            allow_nograd = self.allow_nograd
        return MAMLpp(
            clone_module(self.module),
            lr=self.lr,
            lrs=clone_named_parameters(self.lrs),
            first_order=first_order,
            allow_unused=allow_unused,
            allow_nograd=allow_nograd,
        )
