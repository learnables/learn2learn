#!/usr/bin/env python3

import torch


class KroneckerTransform(object):
    """
    docstring for KroneckerTransform

    TODO:
        * Add the way to compute n and m.
    """

    def __init__(self, transform, bias=False, cholesky=True):
        self.transform = transform
        self.bias = bias
        self.cholesky = cholesky

    def __call__(self, param):
        """docstring for forward"""
        n = None
        m = None
        return self.transform(n, m, bias=self.bias, cholesky=self.cholesky)
