#!/usr/bin/env python3

import torch


class ParameterUpdate(torch.nn.Module):
    """
    Computes updates for for each parameter in `parameters` independently.

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
        self.tranforms_modules = torch.nn.ModuleList(transform_modules)
        self.transforms_indices = transforms_indices

    def forward(
            self,
            loss,
            parameters,
            create_graph=False,
            retain_graph=False,
            ):
        """docstring for forward"""
        gradients = torch.autograd.grad(loss,
                                        parameters,
                                        create_graph=create_graph,
                                        retain_graph=retain_graph)
        updates = []
        for g, t in zip(gradients, self.transforms_indices):
            if t is None:
                update = g
            else:
                transform = self.tranforms_modules[t]
                update = transform(g)
            updates.append(update)
        return updates
