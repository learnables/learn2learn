#!/usr/bin/env python3

import torch as th
from torch import nn


def kronecker_addmm(mat1, mat2, mat3, bias=None, alpha=1.0, beta=1.0):
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/models.py)

    **Description**

    Efficiently computes the matrix-vector product of a large matrix expressed as the Kronecker product of smaller matrices.

    Concretely, the method takes advantage of the following identity:

    $$ \\alpha (A^\\top \\otimes B) \\cdot \\text{vec}(C) + \\beta \\cdot b = \\alpha \\text{vec}(BCA) + \\beta \\cdot b, $$

    where $A \\in \\mathbb{R}^{N \\times N}, B \\in \\mathbb{R}^{M \\times M}, C \\in \\mathbb{R}^{N \\times M}$ and $b$ is a bias term.

    **References**

    1. Arnold et al. 2019. "Decoupling Adaptation from Modeling with Meta-Optimizers for Meta Learning". ArXiv.

    **Arguments**

    * **mat1** (Tensor) - The right Kronecker-factor. ($B$ above)
    * **mat2** (Tensor) - The left Kronecker-factor. ($A$ above)
    * **mat3** (Tensor) - The vector, in matrix form. ($C$ above)
    * **bias** (Tensor, *optional*, default=None) - The bias, in matrix form. ($b$ above)
    * **alpha** (float, *optional*, default=1.0) - The weight of the matrix-vector product. ($\\alpha$ above)
    * **beta** (float, *optional*, default=1.0) - The weight of the bias term. ($\\beta$ above)

    **Example**
    ~~~python
    mat1 = th.randn(5, 5)
    mat2 = th.randn(3, 3)
    mat3 = th.randn(5, 3)
    bias = th.randn(5, 3)
    product = kronecker_addmm(mat1, mat2, mat3, bias)
    ~~~
    """
    res = mat1 @ mat3 @ mat2
    res.mul_(alpha)
    if bias is not None:
        res.add_(beta, bias)
    return res


def cholesky_addmm(mat1, mat2, bias=None, alpha=1.0, beta=1.0):
    """
    Returns alpha * mat1 @ mat1.t() @ mat2 + beta * bias.
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/models.py)

    **Description**

    Efficiently computes the matrix-matrix product of the (possibly low-rank) Cholesky decomposition of a large matrix with another one.

    **References**

    1. Arnold et al. 2019. "Decoupling Adaptation from Modeling with Meta-Optimizers for Meta Learning". ArXiv.

    **Arguments**

    * **mat1** (Tensor) - The Cholesky factor.
    * **mat2** (Tensor) - The second matrix.
    * **bias** (Tensor) - The bias term.
    * **alpha** (float, *optional*, default=1.0) - The weight of the matrix-matrix product.
    * **beta** (float, *optional*, default=1.0) - The weight of the bias.

    **Example**
    ~~~python
    ~~~
    """
    result = mat1 @ (mat1.t() @ mat2)
    result.mul_(alpha)
    if bias is not None:
        result += beta * bias
    return result


class KroneckerLinear(nn.Module):
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/models.py)

    **Description**

    Learns a Kronecker factorization of A@x + b, assuming that:
    * x is in R^{n, m},
    * A is in R^{nm, nm}
    * b is in R^{nm, 1}

    A = self.right.() X self.left is initialized as the identity,
    b = self.bias as a vector of 0's.

    **References**

    1. Arnold et al. 2019. "Decoupling Adaptation from Modeling with Meta-Optimizers for Meta Learning". ArXiv.

    **Arguments**

    * **input_size** (int) - The dimensionality of the input.

    **Example**
    ~~~python
    ~~~
    """
    def __init__(self, n, m, bias=True, cholesky=False, device=None):
        super(KroneckerLinear, self).__init__()
        self.left = nn.Parameter(th.eye(n, device=device))
        self.right = nn.Parameter(th.eye(m, device=device))
        self.bias = None
        self.cholesky = cholesky
        if bias:
            self.bias = nn.Parameter(th.zeros(n, m, device=device))
        self.device = device
        self.to(device=device)

    def forward(self, x):
        old_device = x.device
        if self.device is not None:
            x = x.to(self.device)
        left = self.left
        right = self.right
        if self.cholesky:
            left = left.t() @ left
            right = right.t() @ right
        if len(x.shape) == 1:
            shape = x.shape
            x = x.view(-1, 1)
            x = kronecker_addmm(left, right, x, self.bias)
            return x.view(*shape).to(old_device)
        x = kronecker_addmm(left, right, x, self.bias)
        return x.to(old_device)


class CholeskyLinear(nn.Module):
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/models.py)

    **Description**

    **References**

    1. Arnold et al. 2019. "Decoupling Adaptation from Modeling with Meta-Optimizers for Meta Learning". ArXiv.

    **Arguments**

    * **input_size** (int) - The dimensionality of the input.

    **Example**
    ~~~python
    ~~~
    """
    def __init__(self, n, m, rank=1, bias=False, device=None):
        super(CholeskyLinear, self).__init__()
        self.size = n * m
        self.device = device
        # # Custom Xavier normal init
        self.L = th.randn(self.size, rank, device=device, requires_grad=True)
        self.L.data *= (2)**0.5 * (1.0 / (self.size))**0.5
        self.L = nn.Parameter(self.L)
        self.bias = None
        if bias:
            self.bias = nn.Parameter(th.zeros(self.size, 1,
                                              requires_grad=True),
                                     device=device)

    def forward(self, x):
        old_device = x.device
        if self.device is not None:
            x = x.to(self.device)
        old_shape = x.shape
        if len(x.shape) == 1:
            x = x.view(self.size, 1)
        else:
            x = x.view(-1, self.size, 1)
        x = cholesky_addmm(self.L, x, self.bias)
        return x.view(old_shape).to(old_device)
