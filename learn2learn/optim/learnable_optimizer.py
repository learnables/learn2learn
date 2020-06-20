#!/usr/bin/env python3

import torch
import learn2learn as l2l
import warnings


class LearnableOptimizer(torch.nn.Module):
    """
    docstring for LearnableOptimizer
    """

    def __init__(self, model, transform, lr=1.0):
        super(LearnableOptimizer, self).__init__()
        assert isinstance(model, torch.nn.Module), \
            'model should inherit from nn.Module.'

        # Keep pointer to model, but don't include in self._modules,
        # self._children, or self._parameters
        self.info = {
            'model': model,
        }

        # Create the transforms
        self.transforms = []
        for name, param in model.named_parameters():
            trans = transform(param)
            self.transforms.append(trans)
        self.transforms = torch.nn.ModuleList(self.transforms)
        self.lr = lr

    def step(self, closure=None):
        """
        docstring for step
        TODO: Do we need to recompute flat_grads for RNNs ? - Write a test.
        """
        model = self.info['model']
        # Ignore warnings as torch 1.5+ warns about accessing .grad of non-leaf
        # variables.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for param, transform in zip(model.parameters(),
                                        self.transforms):
                    if hasattr(param, 'grad') and param.grad is not None:
                        # 1. compute update
                        grad = param.grad.detach()
                        grad.requires_grad = False
                        update = - self.lr * transform(grad)

                        # 2. detach parameters
                        param.detach_()
                        param.requires_grad = False
                        param.update = update

            # 3. apply update so that it's differentiable
            l2l.meta_update(model, updates=None)

            for param in model.parameters():
                # 4. retain grad for next update
                param.retain_grad()

    def zero_grad(self):
        """Only reset target parameters."""
        model = self.info['model']
        for p in model.parameters():
            if hasattr(p, 'grad') and p.grad is not None:
                # Do not reset in-place:
                # it breaks the computation graph of step().
                p.grad = torch.zeros_like(p.data)
