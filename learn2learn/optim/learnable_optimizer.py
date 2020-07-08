#!/usr/bin/env python3

import torch
import learn2learn as l2l
import warnings


class LearnableOptimizer(torch.nn.Module):
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/optim/learnable_optimizer.py)

    **Description**

    A PyTorch Optimizer with learnable transform, enabling the implementation
    of meta-descent / hyper-gradient algorithms.

    This optimizer takes a Module and a gradient transform.
    At each step, the gradient of the module is passed through the transforms,
    and the module differentiably update -- i.e. when the next backward is called,
    gradients of both the module and the transform are computed.
    In turn, the transform can be updated via your favorite optmizer.

    **Arguments**

    * **model** (Module) - Module to be updated.
    * **transform** (Module) - Transform used to compute updates of the model.
    * **lr** (float) - Learning rate.

    **References**

    1. Sutton. 1992. “Gain Adaptation Beats Least Squares.”
    2. Schraudolph. 1999. “Local Gain Adaptation in Stochastic Gradient Descent.”
    3. Baydin et al. 2017. “Online Learning Rate Adaptation with Hypergradient Descent.”
    4. Majumder et al. 2019. “Learning the Learning Rate for Gradient Descent by Gradient Descent.”
    5. Jacobsen et al. 2019. “Meta-Descent for Online, Continual Prediction.”

    **Example**

    ~~~python
    linear = nn.Linear(784, 10)
    transform = l2l.optim.ModuleTransform(torch.nn.Linear)
    metaopt = l2l.optim.LearnableOptimizer(linear, transform, lr=0.01)
    opt = torch.optim.SGD(metaopt.parameters(), lr=0.001)

    metaopt.zero_grad()
    opt.zero_grad()
    error = loss(linear(X), y)
    error.backward()
    opt.step()  # update metaopt
    metaopt.step()  # update linear
    ~~~
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
        # TODO: Do we need to recompute flat_grads for RNNs ? - Write a test.
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
            l2l.update_module(model, updates=None)

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
