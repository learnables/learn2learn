#!/usr/bin/env python3

import torch
import learn2learn as l2l


def get_kronecker_dims(param):
    shape = param.shape
    if len(shape) == 2:
        n, m = shape
    elif len(shape) == 1:
        n, m = shape[0], 1
    elif len(shape) == 4:
        n = shape[1]
        m = shape[2] * shape[3]
    return n, m


class KroneckerTransform(object):
    """
    docstring for KroneckerTransform

    TODO:
        * Add the way to compute n and m.
    """

    def __init__(self, transform, bias=False, psd=True):
        self.transform = transform
        self.bias = bias
        self.psd = psd

    def __call__(self, param):
        """docstring for forward"""
        n, m = get_kronecker_dims(param)
        transform = self.transform(
            n=n,
            m=m,
            bias=self.bias,
            psd=self.psd,
        )
        return l2l.optim.transforms.ReshapedTransform(
            transform=transform,
            shape=(-1, n, m)
        )
