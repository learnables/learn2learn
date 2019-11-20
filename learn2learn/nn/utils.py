#!/usr/bin/env python3


def set_gradients(parameters, gradients):
    parameters = list(parameters)
    assert len(parameters) == len(gradients), \
        'parameters and gradients do not have identical lengths.'
    for p, g in zip(parameters, gradients):
        p.grad = g
