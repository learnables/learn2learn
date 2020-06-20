#!/usr/bin/env python3

import torch
import traceback


class ParameterUpdate(torch.nn.Module):
    """
    Computes updates for each parameter in `parameters` independently.

    As we wish to support some transforms being None, we need to keep track of
    the index at which the transfors are instantiated.
    We can't keep two lists of transforms, as only the ModuleList will be
    updated with clone_module.
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
        """docstring for forward"""
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
