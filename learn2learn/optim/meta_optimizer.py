#!/usr/bin/env python3

import torch
from torch.optim import Optimizer


def _identity_update(grad):
    return grad


def _symbolic_param_update(module, restrict=None):
    if restrict is None:
        restrict = list(module.parameters())

    # Update the params
    for param_key in module._parameters:
        p = module._parameters[param_key]
        if p is not None and id(p) in restrict and p.grad is not None:
            module._parameters[param_key] = p - p.grad

    # Then, recurse for each submodule
    for module_key in module._modules:
        _symbolic_param_update(module._modules[module_key], restrict=restrict)


class MetaOptimizer(Optimizer):

    """
    Doc goes here.
    """

    def __init__(self, params, model, update=None, create_graph=False):
        if update is None:
            update = _identity_update
        defaults = {
            'update': update,
            'create_graph': create_graph
        }
        super(MetaOptimizer, self).__init__(params, defaults)
        self.model = model

    def step(self, closure=None):
        update_params = []
        for group in self.param_groups:
            update = group['update']
            create_graph = group['create_graph']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if create_graph:
                    updt = update(grad)
                else:
                    # Detach both param and grad from the existing graph.
                    grad = grad.detach()
                    grad.requires_grad = False
                    updt = update(grad)
                    p.detach_()
                    p.requires_grad = False
                p.grad = updt
                update_params.append(id(p))
        self._apply_update(update_params)

    def _apply_update(self, update_params):
        # Apply the update to the model.
        # NOTE: We need to keep track of old param to update param groups and states.
        # Otherwise, the references in both point to the outdated param tensor.
        old_params = list(self.model.parameters())
        _symbolic_param_update(self.model, restrict=update_params)
        self._update_references(old_params, update_params)

    def _update_references(self, old_params, update_params):
        # Update the reference of param_states to the new params
        for op, p in zip(old_params, self.model.parameters()):
            if id(op) in update_params:
                p.retain_grad()
                self.state[p] = self.state[op]
                del self.state[op]
            else:
                import pdb; pdb.set_trace()

        # Update the reference of param_groups to the new params
        old_params = [id(p) for p in old_params]
        new_params = list(self.model.parameters())
        for group in self.param_groups:
            new_group = []
            for p in group['params']:
                idx = old_params.index(id(p))
                new_group.append(new_params[idx])
            del group['params']
            group['params'] = new_group


    def parameters(self):
        for group in self.param_groups:
            update = group['update']
            if hasattr(update, 'parameters'):
                for p in update.parameters():
                    yield p
