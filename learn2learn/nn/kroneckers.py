#!/usr/bin/env python3

import torch as th
from torch import nn


def kronecker_addmm(mat1, mat2, mat3, bias=None, alpha=1.0, beta=1.0):
    """
    Returns alpha * (mat2.t() X mat1) @ vec(mat3) + beta * vec(bias)
    (Assuming bias is not None.)
    """
    res = mat1 @ mat3 @ mat2
    res.mul_(alpha)
    if bias is not None:
        res.add_(alpha=beta, other=bias)
    return res


class KroneckerLinear(nn.Module):

    r"""
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/nn/kroneckers.py)

    **Description**

    A linear transformation whose parameters are expressed as a Kronecker product.

    This Module maps an input vector \(x \in \mathbb{R}^{nm} \) to \(y = Ax + b\) such that:

    \[
    A = R^\top \otimes L,
    \]

    where \(L \in \mathbb{R}^{n \times n}\) and \(R \in \mathbb{R}^{m \times m}\) are the learnable Kronecker factors.
    This implementation can reduce the memory requirement for large linear mapping
    from \(\mathcal{O}(n^2 \cdot m^2)\) to \(\mathcal{O}(n^2 + m^2)\), but forces \(y \in \mathbb{R}^{nm}\).

    The matrix \(A\) is initialized as the identity, and the bias as a zero vector.

    **Arguments**

    * **n** (int) - Dimensionality of the left Kronecker factor.
    * **m** (int) - Dimensionality of the right Kronecker factor.
    * **bias** (bool, *optional*, default=True) - Whether to include the bias term.
    * **psd** (bool, *optional*, default=False) - Forces the matrix \(A\) to be positive semi-definite if True.
    * **device** (device, *optional*, default=None) - The device on which to instantiate the Module.

    **References**

    1. Jose et al. 2018. "Kronecker recurrent units".
    2. Arnold et al. 2019. "When MAML can adapt fast and how to assist when it cannot".

    **Example**
    ~~~python
    m, n = 2, 3
    x = torch.randn(6)
    kronecker = KroneckerLinear(n, m)
    y = kronecker(x)
    y.shape  # (6, )
    ~~~
    """

    def __init__(self, n, m, bias=True, psd=False, device=None):
        super(KroneckerLinear, self).__init__()
        self.left = nn.Parameter(th.eye(n, device=device))
        self.right = nn.Parameter(th.eye(m, device=device))
        self.bias = None
        self.psd = psd
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
        if self.psd:
            left = left.t() @ left
            right = right.t() @ right
        n = self.left.size(0)
        m = self.right.size(0)
        if len(x.shape) == 1:
            if x.size(0) != n * m:
                raise ValueError("Input vector must have size n*m")
            X = x.reshape(m, n).t()
            Y = kronecker_addmm(left, right, X, self.bias)
            y = Y.t().flatten()
            return y.to(old_device)
        if x.shape[-2:] != (n, m):
            raise ValueError(
                "Final two dimensions of input tensor must have shape (n, m)"
            )
        x = kronecker_addmm(left, right, x, self.bias)
        return x.to(old_device)


class KroneckerRNN(nn.Module):

    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/nn/kroneckers.py)

    **Description**

    Implements a recurrent neural network whose matrices are parameterized via their Kronecker factors.
    (See `KroneckerLinear` for details.)

    **Arguments**

    * **n** (int) - Dimensionality of the left Kronecker factor.
    * **m** (int) - Dimensionality of the right Kronecker factor.
    * **bias** (bool, *optional*, default=True) - Whether to include the bias term.
    * **sigma** (callable, *optional*, default=None) - The activation function.

    **References**

    1. Jose et al. 2018. "Kronecker recurrent units".

    **Example**
    ~~~python
    m, n = 2, 3
    x = torch.randn(6)
    h = torch.randn(6)
    kronecker = KroneckerRNN(n, m)
    y, new_h = kronecker(x, h)
    y.shape  # (6, )
    ~~~
    """

    def __init__(self, n, m, bias=True, sigma=None):
        super(KroneckerRNN, self).__init__()
        self.W_h = KroneckerLinear(n, m, bias=bias)
        self.U_h = KroneckerLinear(n, m, bias=bias)
        self.W_y = KroneckerLinear(n, m, bias=bias)

        if sigma is None:
            sigma = nn.Tanh()
        self.sigma = sigma

    def forward(self, x, hidden):
        new_hidden = self.W_h(x) + self.U_h(hidden)
        new_hidden = self.sigma(new_hidden)
        output = self.W_y(new_hidden)
        return output, new_hidden


class KroneckerLSTM(nn.Module):

    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/nn/kroneckers.py)

    **Description**

    Implements an LSTM using a factorization similar to the one of
    `KroneckerLinear`.

    **Arguments**

    * **n** (int) - Dimensionality of the left Kronecker factor.
    * **m** (int) - Dimensionality of the right Kronecker factor.
    * **bias** (bool, *optional*, default=True) - Whether to include the bias term.
    * **sigma** (callable, *optional*, default=None) - The activation function.

    **References**

    1. Jose et al. 2018. "Kronecker recurrent units".

    **Example**
    ~~~python
    n, m = 2, 3
    x = torch.randn(n, m)
    h = torch.randn(n, m)
    c = torch.zeros(n, m)
    kronecker = KroneckerLSTM(n, m)
    y, new_h = kronecker(x, (h, c))
    y.shape  # (2, 3)
    ~~~
    """

    def __init__(self, n, m, bias=True, sigma=None):
        super(KroneckerLSTM, self).__init__()
        self.W_ii = KroneckerLinear(n, m, bias=bias)
        self.W_hi = KroneckerLinear(n, m, bias=bias)
        self.W_if = KroneckerLinear(n, m, bias=bias)
        self.W_hf = KroneckerLinear(n, m, bias=bias)
        self.W_ig = KroneckerLinear(n, m, bias=bias)
        self.W_hg = KroneckerLinear(n, m, bias=bias)
        self.W_io = KroneckerLinear(n, m, bias=bias)
        self.W_ho = KroneckerLinear(n, m, bias=bias)
        if sigma is None:
            sigma = nn.Sigmoid()
        self.sigma = sigma
        self.tanh = nn.Tanh()

    def forward(self, x, hidden):
        h, c = hidden
        i = self.sigma(self.W_ii(x) + self.W_hi(h))
        f = self.sigma(self.W_if(x) + self.W_hf(h))
        g = self.tanh(self.W_ig(x) + self.W_hg(h))
        o = self.sigma(self.W_io(x) + self.W_ho(h))
        c = f * c + i * g
        h = o * self.tanh(c)
        return h, (h, c)
