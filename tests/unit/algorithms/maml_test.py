#!/usr/bin/env python3

import unittest
import torch as th
import learn2learn as l2l

NUM_INPUTS = 7
INPUT_SIZE = 10
HIDDEN_SIZE = 20
INNER_LR = 0.1
EPSILON = 1e-8


def close(x, y):
    return (x-y).norm(p=2) <= EPSILON


class TestMAMLAlgorithm(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_clone_module(self):
        for first_order in [False, True]:
            model = th.nn.Sequential(th.nn.Linear(INPUT_SIZE, HIDDEN_SIZE),
                                     th.nn.ReLU(),
                                     th.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
                                     th.nn.Sigmoid(),
                                     th.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
                                     th.nn.Softmax())
            maml = l2l.algorithms.MAML(model,
                                       lr=INNER_LR,
                                       first_order=first_order)
            X = th.randn(NUM_INPUTS, INPUT_SIZE)
            ref = model(X)
            for clone in [maml.clone(), maml.clone()]:
                out = clone(X)
                self.assertTrue(close(ref, out))

    def test_graph_connection(self):
        model = th.nn.Sequential(th.nn.Linear(INPUT_SIZE, HIDDEN_SIZE),
                                 th.nn.ReLU(),
                                 th.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
                                 th.nn.Sigmoid(),
                                 th.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
                                 th.nn.Softmax())
        maml = l2l.algorithms.MAML(model,
                                   lr=INNER_LR,
                                   first_order=False)
        X = th.randn(NUM_INPUTS, INPUT_SIZE)
        ref = maml(X)
        clone = maml.clone()
        out = clone(X)
        out.norm(p=2).backward()
        for p in model.parameters():
            self.assertTrue(hasattr(p, 'grad'))
            self.assertTrue(p.grad.norm(p=2).item() > 0.0)


if __name__ == '__main__':
    unittest.main()
