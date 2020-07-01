#!/usr/bin/env python3

import torch
import traceback


class ParameterUpdate(torch.nn.Module):
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/optim/parameter_update.py)

    **Description**

    Convenience class to implement custom update functions.

    Objects instantiated from this class behave similarly to `torch.autograd.grad`,
    but return parameter updates as opposed to gradients.
    Concretely, the gradients are first computed, then fed to their respective transform
    whose output is finally returned to the user.

    Additionally, this class supports parameters that might not require updates by setting
    the `allow_nograd` flag to True.
    In this case, the returned update is `None`.

    **Arguments**

    * **parameters** (list) - Parameters of the model to update.
    * **transform** (callable) - A callable that returns an instantiated
        transform given a parameter.

    **Example**
    ~~~python
    model = torch.nn.Linear()
    transform = l2l.optim.KroneckerTransform(l2l.nn.KroneckerLinear)
    get_update = ParameterUpdate(model, transform)
    opt = torch.optim.SGD(model.parameters() + get_update.parameters())

    for iteration in range(10):
        opt.zero_grad()
        error = loss(model(X), y)
        updates = get_update(
            error,
            model.parameters(),
            create_graph=True,
        )
        l2l.update_module(model, updates)
        opt.step()
    ~~~
    """

    def __init__(self, parameters, transform):
        super(ParameterUpdate, self).__init__()
        transforms_indices = []
        transform_modules = []
        module_counter = 0
        for param in parameters:
            t = transform(param)
            if t is None:
                idx = None
            elif isinstance(t, torch.nn.Module):
                transform_modules.append(t)
                idx = module_counter
                module_counter += 1
            else:
                msg = 'Transform should be either a Module or None.'
                raise ValueError(msg)
            transforms_indices.append(idx)
        self.transforms_modules = torch.nn.ModuleList(transform_modules)
        self.transforms_indices = transforms_indices

    def forward(
            self,
            loss,
            parameters,
            create_graph=False,
            retain_graph=False,
            allow_unused=False,
            allow_nograd=False,
            ):
        """
        **Description**

        Similar to torch.autograd.grad, but passes the gradients through the
        provided transform.

        **Arguments**

        * **loss** (Tensor) - The loss to differentiate.
        * **parameters** (iterable) - Parameters w.r.t. which we want to compute the update.
        * **create_graph** (bool, *optional*, default=False) - Same as `torch.autograd.grad`.
        * **retain_graph** (bool, *optional*, default=False) - Same as `torch.autograd.grad`.
        * **allow_unused** (bool, *optional*, default=False) - Same as `torch.autograd.grad`.
        * **allow_nograd** (bool, *optional*, default=False) - Properly handles parameters
            that do not require gradients. (Their update will be `None`.)

        """
        updates = []
        if allow_nograd:
            parameters = list(parameters)
            diff_params = [p for p in parameters if p.requires_grad]
            grad_params = torch.autograd.grad(
                loss,
                diff_params,
                retain_graph=create_graph,
                create_graph=create_graph,
                allow_unused=allow_unused)
            gradients = []

            # Handles gradients for non-differentiable parameters
            grad_counter = 0
            for param in parameters:
                if param.requires_grad:
                    gradient = grad_params[grad_counter]
                    grad_counter += 1
                else:
                    gradient = None
                gradients.append(gradient)
        else:
            try:
                gradients = torch.autograd.grad(
                    loss,
                    parameters,
                    create_graph=create_graph,
                    retain_graph=retain_graph,
                    allow_unused=allow_unused,
                )
            except RuntimeError:
                traceback.print_exc()
                msg = 'learn2learn: Maybe try with allow_nograd=True and/or' +\
                      'allow_unused=True ?'
                print(msg)
        for g, t in zip(gradients, self.transforms_indices):
            if t is None or g is None:
                update = g
            else:
                transform = self.transforms_modules[t]
                update = transform(g)
            updates.append(update)
        return updates
