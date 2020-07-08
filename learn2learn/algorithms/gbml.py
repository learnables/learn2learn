#!/usr/bin/env python3

import torch
import learn2learn as l2l


class GBML(torch.nn.Module):
    """

    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/algorithms/gbml.py)

    **Description**

    General wrapper for gradient-based meta-learning implementations.

    A variety of algorithms can simply be implemented by changing the kind
    of `transform` used during fast-adaptation.
    For example, if the transform is `Scale` we recover Meta-SGD [2] with `adapt_transform=False`
    and Alpha MAML [4] with `adapt_transform=True`.
    If the transform is a Kronecker-factored module (e.g. neural network, or linear), we recover
    KFO from [5].

    **Arguments**

    * **module** (Module) - Module to be wrapped.
    * **tranform** (Module) - Transform used to update the module.
    * **lr** (float) - Fast adaptation learning rate.
    * **adapt_transform** (bool, *optional*, default=False) - Whether to update the transform's
        parameters during fast-adaptation.
    * **first_order** (bool, *optional*, default=False) - Whether to use the first-order
        approximation.
    * **allow_unused** (bool, *optional*, default=None) - Whether to allow differentiation
        of unused parameters. Defaults to `allow_nograd`.
    * **allow_nograd** (bool, *optional*, default=False) - Whether to allow adaptation with
        parameters that have `requires_grad = False`.

    **References**

    1. Finn et al. 2017. “Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks.”
    2. Li et al. 2017. “Meta-SGD: Learning to Learn Quickly for Few-Shot Learning.”
    3. Park & Oliva. 2019. “Meta-Curvature.”
    4. Behl et al. 2019. “Alpha MAML: Adaptive Model-Agnostic Meta-Learning.”
    5. Arnold et al. 2019. “When MAML Can Adapt Fast and How to Assist When It Cannot.”

    **Example**

    ~~~python
    model = SmallCNN()
    transform = l2l.optim.ModuleTransform(torch.nn.Linear)
    gbml = l2l.algorithms.GBML(
        module=model,
        transform=transform,
        lr=0.01,
        adapt_transform=True,
    )
    gbml.to(device)
    opt = torch.optim.SGD(gbml.parameters(), lr=0.001)

    # Training with 1 adaptation step
    for iteration in range(10):
        opt.zero_grad()
        task_model = gbml.clone()
        loss = compute_loss(task_model)
        task_model.adapt(loss)
        loss.backward()
        opt.step()
    ~~~
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
        **Description**

        Similar to `MAML.clone()`.

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
        **Description**

        Takes a gradient step on the loss and updates the cloned parameters in place.

        The parameters of the transform are only adapted if `self.adapt_update` is `True`.

        **Arguments**

        * **loss** (Tensor) - Loss to minimize upon update.
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
