#!/usr/bin/env python3

import torch
import learn2learn as l2l


class GBML(torch.nn.Module):
    """

    TODO:
        * Support for allow_unused
        * Support for allow_nograd
    """

    def __init__(
            self,
            module,
            transform,
            lr=1.0,
            adapt_transform=False,
            first_order=False,
            allow_unused=False,
            allow_nograd=False,
            **kwargs,
            ):
        super(GBML, self).__init__()
        self.module = module
        self.transform = transform
        self.adapt_transform = adapt_transform
        self.lr = lr
        self.first_order = first_order
        self.allow_unused = allow_unused
        self.allow_nograd = allow_nograd
        if 'compute_update' in kwargs:
            self.compute_update = kwargs.get('compute_update')
        else:
            self.compute_update = l2l.optim.ParameterUpdate(
                    parameters=self.module.parameters(),
                    transform=transform,
            )
        self.diff_sgd = l2l.optim.DifferentiableSGD(lr=self.lr)
        # Whether the module params have already been updated with the
        # updates from compute_update. Used to keep track of whether we
        # can compute the gradient of compute_update's parameters.
        self._params_updated = False

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def clone(
            self,
            first_order=None,
            allow_unused=None,
            allow_nograd=None,
            adapt_transform=None,
            ):
        """
        docs
        """
        if first_order is None:
            first_order = self.first_order
        if allow_unused is None:
            allow_unused = self.allow_unused
        if allow_nograd is None:
            allow_nograd = self.allow_nograd
        if adapt_transform is None:
            adapt_transform = self.adapt_transform
        module_clone = l2l.clone_module(self.module)
        update_clone = l2l.clone_module(self.compute_update)
        return GBML(
            module=module_clone,
            transform=self.transform,
            lr=self.lr,
            adapt_transform=adapt_transform,
            first_order=first_order,
            allow_unused=allow_unused,
            allow_nograd=allow_nograd,
            compute_update=update_clone,
        )

    def adapt(
            self,
            loss,
            first_order=None,
            allow_nograd=None,
            allow_unused=None,
            ):
        """
        docs
        """
        if first_order is None:
            first_order = self.first_order
        if allow_unused is None:
            allow_unused = self.allow_unused
        if allow_nograd is None:
            allow_nograd = self.allow_nograd
        second_order = not first_order

        if self.adapt_transform and self._params_updated:
            # Update the learnable update function
            update_grad = torch.autograd.grad(
                loss,
                self.compute_update.parameters(),
                create_graph=second_order,
                retain_graph=second_order,
                allow_unused=allow_unused,
            )
            self.diff_sgd(self.compute_update, update_grad)
            self._params_updated = False

        # Update the module
        updates = self.compute_update(
            loss,
            self.module.parameters(),
            create_graph=second_order or self.adapt_transform,
            retain_graph=second_order or self.adapt_transform,
            allow_unused=allow_unused,
            allow_nograd=allow_nograd,
        )
        self.diff_sgd(self.module, updates)
        self._params_updated = True
