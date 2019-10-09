#!/usr/bin/env python3

import torch
from torch.optim import Optimizer
from torch.optim.optimizer import required


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
        self.model = model
        if update is None:
            update = _identity_update
        defaults = {
            'update': update,
            'create_graph': create_graph
        }
        super(MetaOptimizer, self).__init__(params, defaults)

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
        """
        old_params: the parameters of self.model before the update.
        update_params: the id of the parameters that were updated.
        """
        ids = [id(p) for p in old_params]
        for up in update_params:
            print(up in ids)
        # Update the reference of param_states to the new params
        for op, p in zip(old_params, self.model.parameters()):
            if id(op) in update_params:
                p.retain_grad()
                self.state[p] = self.state[op]
                del self.state[op]
#            else:
#                import pdb; pdb.set_trace()

        # Update the reference of param_groups to the new params
        old_params = [id(p) for p in old_params]
        new_params = list(self.model.parameters())
        for group in self.param_groups:
            new_group = []
            for p in group['params']:
                try:
                    idx = old_params.index(id(p))
                except:
                    import pdb; pdb.set_trace()
                new_group.append(new_params[idx])
            del group['params']
            group['params'] = new_group


    def parameters(self):
        for group in self.param_groups:
            update = group['update']
            if hasattr(update, 'parameters'):
                for p in update.parameters():
                    yield p

    def add_param_group(self, param_group):
        assert isinstance(param_group, dict), "param group must be a dict"

        params = param_group['params']
        if isinstance(params, torch.Tensor):
            param_group['params'] = [params]
        elif isinstance(params, set):
            raise TypeError('optimizer parameters need to be organized in ordered collections, but '
                            'the ordering of tensors in sets will change between runs. Please use a list instead.')
        else:
            param_group['params'] = list(params)

        create_graph = self.defaults['create_graph']
        for param in param_group['params']:
            if not isinstance(param, torch.Tensor):
                raise TypeError("optimizer can only optimize Tensors, "
                                "but one of the params is " + torch.typename(param))
            if not create_graph and not param.is_leaf:
                raise ValueError("can't optimize a non-leaf Tensor")

        for name, default in self.defaults.items():
            if default is required and name not in param_group:
                raise ValueError("parameter group didn't specify a value of required optimization parameter " +
                                 name)
            else:
                param_group.setdefault(name, default)

        param_set = set()
        for group in self.param_groups:
            param_set.update(set(group['params']))

        if not param_set.isdisjoint(set(param_group['params'])):
            raise ValueError("some parameters appear in more than one parameter group")

        self.param_groups.append(param_group)
