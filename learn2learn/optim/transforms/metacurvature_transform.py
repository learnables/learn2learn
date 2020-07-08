#!/usr/bin/env python3

import torch
import numpy as np


class MetaCurvatureTransform(torch.nn.Module):

    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/optim/transforms/module_transform.py)

    **Description**

    Implements the Meta-Curvature transform of Park and Oliva, 2019.

    Unlike `ModuleTranform` and `KroneckerTransform`, this class does not wrap other Modules but is directly
    called on a weight to instantiate the transform.

    **Arguments**

    * **param** (Tensor) - The weight whose gradients will be transformed.
    * **lr** (float, *optional*, default=1.0) - Scaling factor of the udpate. (non-learnable)

    **References**

    1. Park & Oliva. 2019. Meta-curvature.

    **Example**
    ~~~python
    classifier = torch.nn.Linear(784, 10, bias=False)
    metacurvature_update = MetaCurvatureTransform(classifier.weight)
    loss(classifier(X), y).backward()
    update = metacurvature_update(classifier.weight.grad)
    classifier.weight.data.add_(-lr, update)  # Not a differentiable update. See l2l.optim.DifferentiableSGD.
    ~~~
    """

    def __init__(self, param, lr=1.0):
        super(MetaCurvatureTransform, self).__init__()
        self.lr = lr
        shape = param.shape
        if len(shape) == 1:  # bias
            self.dim = 1
            self.mc = torch.nn.Parameter(torch.ones_like(param))
        elif len(shape) == 2:  # FC
            self.dim = 2
            self.mc_in = torch.nn.Parameter(torch.eye(shape[0]))
            self.mc_out = torch.nn.Parameter(torch.eye(shape[1]))
        elif len(shape) == 4:  # CNN
            self.dim = 4
            self.n_in = shape[0]
            self.n_out = shape[1]
            self.n_f = int(np.prod(shape) / (self.n_in * self.n_out))
            self.mc_in = torch.nn.Parameter(torch.eye(self.n_in))
            self.mc_out = torch.nn.Parameter(torch.eye(self.n_out))
            self.mc_f = torch.nn.Parameter(torch.eye(self.n_f))
        else:
            raise NotImplementedError('Parameter with shape',
                                      shape,
                                      'is not supported by MetaCurvature.')

    def forward(self, grad):
        if self.dim == 1:
            update = self.mc * grad
        elif self.dim == 2:
            update = self.mc_in @ grad @ self.mc_out
        else:
            # Following the ref. implementation, we use TensorFlow's shapes
            # TODO: Rewrite for PyTorch's conv and avoid contiguous()/permute()
            update = grad.permute(2, 3, 0, 1).contiguous()
            shape = update.shape
            update = update.view(-1, self.n_out) @ self.mc_out
            update = self.mc_f @ update.view(self.n_f, -1)
            update = update.view(self.n_f, self.n_in, self.n_out)
            update = update.permute(1, 0, 2).contiguous().view(self.n_in, -1)
            update = self.mc_in @ update
            update = update.view(
                    self.n_in,
                    self.n_f,
                    self.n_out).permute(1, 0, 2).contiguous().view(shape)
            update = update.permute(2, 3, 0, 1).contiguous()
        return self.lr * update
