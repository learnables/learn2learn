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
        res.add_(beta, bias)
    return res


class KroneckerLinear(nn.Module):

    """
    Learns a Kronecker factorization of A@x + b, assuming that:
    * x is in R^{n, m},
    * A is in R^{nm, nm}
    * b is in R^{nm, 1}

    A = self.right.() X self.left is initialized as the identity,
    b = self.bias as a vector of 0's.
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
        if len(x.shape) == 1:
            shape = x.shape
            x = x.view(-1, 1)
            x = kronecker_addmm(left, right, x, self.bias)
            return x.view(*shape).to(old_device)
        x = kronecker_addmm(left, right, x, self.bias)
        return x.to(old_device)


class KroneckerRNN(nn.Module):

    """
    Computes

    h = sigma(W_h@x + b_x + U_h@h + b_h)
    y = W_y@h + b_y

    assuming a similar decomposition as for the KroneckerLinear.
    sigma is a nn.Tanh() by default.
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
    Implements an LSTM using a decomposition similar to the one of
    KroneckerLinear.
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
