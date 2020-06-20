#!/usr/bin/env python3

import torch
import learn2learn as l2l


class GBML(torch.nn.Module):
    """

    TODO:
        * Support for first_order
        * Support for allow_unused
        * Support for allow_nograd
    """

    def __init__(
            self,
            module,
            transform=None,
            lr=1.0,
            adapt_transform=False,
            **kwargs,
            ):
        super(GBML, self).__init__()
        self.module = module
        self.transform = transform
        self.adapt_transform = adapt_transform
        self.lr = lr
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

    def clone(self):
        """
        docs
        """
        module_clone = l2l.clone_module(self.module)
        update_clone = l2l.clone_module(self.compute_update)
        return GBML(
            module=module_clone,
            transform=self.transform,
            lr=self.lr,
            adapt_transform=self.adapt_transform,
            compute_update=update_clone,
        )

    def adapt(self, loss):
        """
        docs
        """
        if self.adapt_transform and self._params_updated:
            # Update the learnable update function
            update_grad = torch.autograd.grad(
                loss,
                self.compute_update.parameters(),
                create_graph=True,
                retain_graph=True)
            self.diff_sgd(self.compute_update, update_grad)
            self._params_updated = False

        # Update the module
        updates = self.compute_update(
            loss,
            self.module.parameters(),
            create_graph=True,
            retain_graph=True)
        self.diff_sgd(self.module, updates)
        self._params_updated = True
