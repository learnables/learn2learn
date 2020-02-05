#!/usr/bin/env python3

import torch
from torch import nn
from torch import autograd
from torch.optim import Optimizer
from torch.optim.optimizer import required

import learn2learn as l2l

from copy import deepcopy


def _identity_update(grad):
    return grad


def _symbolic_param_update(module, restrict=None):
    if isinstance(module, nn.Module):
        if restrict is None:
            restrict = list(module.parameters())

        # Update the params
        for param_key in module._parameters:
            p = module._parameters[param_key]
            if p is not None and id(p) in restrict and p.grad is not None:
                module._parameters[param_key] = p - p.grad

        # Then, recurse for each submodule
        for module_key in module._modules:
            _symbolic_param_update(module._modules[module_key],
                                   restrict=restrict)


class MetaOptimizer(Optimizer):
    """

    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/optim/meta_optimizer.py)

    **Description**

    General class to implement many meta-optimization algorithms.

    This class can be used for supervised learning to implement Sutton, Schroedolph, etc...

    It can also be used for few-shot meta-learning to implement MAML, MetaSGD, MetaCurvature, KFC, etc...

    MetaOptimizer strives to behave like a standard optimizer: you can pass parameter groups, and associate an update function
    to each group.

    The current implementation requires that params == model.parameters(). (Might change in the future.)

    **Arguments**

    * **params** (iterable) - The parameters of the module to be optimized.
    * **model** (nn.Module) - Which model uses those parameters.
    * **update** (callable, *optional*, default=None) - The function called to compute updates.
    * **create_graph** (bool, *optional*, default=False) - Whether to allow differentiation of the update steps.

    **References**

    1. Jacobs. 1988. "Improved Delta-Bar-Delta".
    2. Sutton. 1992. "Incremental Delta-Bar-Delta".
    3. Shroedolph. 1999. ""
    4. Baydin et al. 2017. "Hypergradient"
    5. Park & Oliva. 2019. "Meta-Curvature"
    5. Arnold et al. 2019. ""

    **Example**

    ~~~python
    ~~~
    """
    def __init__(self, params, model, update=None, create_graph=False):
        self.model = model
        self.create_graph = create_graph
        if update is None:
            update = _identity_update
        defaults = {'update': update, 'create_graph': create_graph}
        super(MetaOptimizer, self).__init__(params, defaults)
        assert len(list(model.parameters())) == sum([len(pg['params']) for pg in self.param_groups], 0), \
            'MetaOptimizers only work on the entire parameter set of the specified model.'
        self.num_parameters = len(list(self.parameters()))

    def clone(self, model=None):
        """
        model is the new model the clone will optimize, typically a clone of the actual model.
        """
        # TODO: It only supports optimizing over the fully range of parameters of the model.
        # NOTE: This implementation is ugly, but tricky to get right.
        # The problem is that you need to clone the optimizer, while keeping the proper references to
        # the parameters in state_dict and param_state.
        # Since the API is tricky to get right, we'll roll with this for now.
        if model is None:
            model = self.model
        # if len(self.param_groups) == 1:
        #     updates = [l2l.clone_module(self.param_groups['update']) for _ in model.parameters()]
        # else:
        #     updates = [l2l.clone_module(pg['update']) for pg in self.param_groups]
        updates = [l2l.clone_module(pg['update'])
                   if isinstance(pg['update'], nn.Module)
                   else deepcopy(pg['update']) for pg in self.param_groups]

        for p in model.parameters():
            p.retain_grad()

        new_opt = MetaOptimizer([{
            'params': [p],
            'update': u
        } for p, u in zip(model.parameters(), updates)],
                                model,
                                create_graph=self.create_graph)
        return new_opt

    def adapt(self, loss, lr=1.0):
        if self.num_parameters:
            # Compute gradients w.r.t opt
            opt_grads = autograd.grad(loss,
                                      self.parameters(),
                                      allow_unused=True,
                                      create_graph=True)
            # Update opt via GD
            l2l.nn.utils.set_gradients(self.parameters(), opt_grads)
            for group in self.param_groups:
                update = group['update']
                group['update'] = l2l.algorithms.maml_update(model=update, lr=lr)

    def step(self, loss=None):
        """
        Compute the next parameter iterate using the provided update functions.
        """
        assert not callable(loss), \
            'loss should not be callable for MetaOptimizers.'

        if loss is not None and self.create_graph:
            # Assumes that for supervised, the gradients are pre-computed.
            assert loss.requires_grad, \
                    'loss does not require grad.'
            # Compute gradients w.r.t. model
            model_params = sum([pg['params'] for pg in self.param_groups], [])
            model_grads = autograd.grad(loss, model_params, create_graph=True)
            # Update learner via opt
            l2l.nn.utils.set_gradients(model_params, model_grads)

        # Update parameters
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
            assert up in ids, \
                    'Failure in MetaOptimizer: somehow, the references to old' \
                    'parameters were not in the list of updated parameters.'
        # Update the reference of param_states to the new params
        for op, p in zip(old_params, self.model.parameters()):
            if id(op) in update_params:
                p.retain_grad()
                self.state[p] = self.state[op]
                del self.state[op]

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
        """
        Returns the parameters of the update function(s).
        """
        for group in self.param_groups:
            update = group['update']
            if hasattr(update, 'parameters'):
                for p in update.parameters():
                    yield p

    def add_param_group(self, param_group):
        # Copied from torch's Optimizer, but added support for create_graph.
        assert isinstance(param_group, dict), "param group must be a dict"

        params = param_group['params']
        if isinstance(params, torch.Tensor):
            param_group['params'] = [params]
        elif isinstance(params, set):
            raise TypeError(
                'optimizer parameters need to be organized in ordered collections, but '
                'the ordering of tensors in sets will change between runs. Please use a list instead.'
            )
        else:
            param_group['params'] = list(params)

        create_graph = self.defaults['create_graph']
        for param in param_group['params']:
            if not isinstance(param, torch.Tensor):
                raise TypeError("optimizer can only optimize Tensors, "
                                "but one of the params is " +
                                torch.typename(param))
            if not create_graph and not param.is_leaf:
                raise ValueError("can't optimize a non-leaf Tensor")

        for name, default in self.defaults.items():
            if default is required and name not in param_group:
                raise ValueError(
                    "parameter group didn't specify a value of required optimization parameter "
                    + name)
            else:
                param_group.setdefault(name, default)

        param_set = set()
        for group in self.param_groups:
            param_set.update(set(group['params']))

        if not param_set.isdisjoint(set(param_group['params'])):
            raise ValueError(
                "some parameters appear in more than one parameter group")

        self.param_groups.append(param_group)

    def zero_grad(self):
        """
        Resets the gradients of both the model and the optimizer.
        """
        super(MetaOptimizer, self).zero_grad()
        for p in self.parameters():
            if hasattr(p, 'grad') and p.grad is not None:
                p.grad = torch.zeros_like(p.data)

    def to(self, *args, **kwargs):
        for pg in self.param_groups:
            update = pg['update']
            if hasattr(update, 'to'):
                update.to(*args, **kwargs)
        return self
