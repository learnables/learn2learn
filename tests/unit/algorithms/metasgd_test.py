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


class TestMetaSGDAlgorithm(unittest.TestCase):

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
            meta = l2l.algorithms.MetaSGD(model,
                                          lr=INNER_LR,
                                          first_order=first_order)
            X = th.randn(NUM_INPUTS, INPUT_SIZE)
            ref = model(X)
            for clone in [meta.clone(), meta.clone()]:
                out = clone(X)
                self.assertTrue(close(ref, out))

    def test_graph_connection(self):
        model = th.nn.Sequential(th.nn.Linear(INPUT_SIZE, HIDDEN_SIZE),
                                 th.nn.ReLU(),
                                 th.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
                                 th.nn.Sigmoid(),
                                 th.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
                                 th.nn.Softmax())
        meta = l2l.algorithms.MetaSGD(model,
                                      lr=INNER_LR,
                                      first_order=False)
        X = th.randn(NUM_INPUTS, INPUT_SIZE)
        ref = meta(X)
        clone = meta.clone()
        out = clone(X)
        out.norm(p=2).backward()
        for p in meta.parameters():
            self.assertTrue(hasattr(p, 'grad'))
            self.assertTrue(p.grad.norm(p=2).item() > 0.0)

    def test_meta_lr(self):
        model = th.nn.Sequential(th.nn.Linear(INPUT_SIZE, HIDDEN_SIZE),
                                 th.nn.ReLU(),
                                 th.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
                                 th.nn.Sigmoid(),
                                 th.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
                                 th.nn.Softmax())
        meta = l2l.algorithms.MetaSGD(model,
                                      lr=INNER_LR,
                                      first_order=False)
        num_params = sum([p.numel() for p in model.parameters()])
        meta_params = sum([p.numel() for p in meta.parameters()])
        self.assertEqual(2 * num_params, meta_params)
        for lr in meta.lrs:
            self.assertTrue(close(lr, INNER_LR))

    def test_adaptation(self):
        model = th.nn.Sequential(th.nn.Linear(INPUT_SIZE, HIDDEN_SIZE),
                                 th.nn.ReLU(),
                                 th.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
                                 th.nn.Sigmoid(),
                                 th.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
                                 th.nn.Softmax())
        meta = l2l.algorithms.MetaSGD(model,
                                      lr=INNER_LR,
                                      first_order=False)
        X = th.randn(NUM_INPUTS, INPUT_SIZE)
        clone = meta.clone()
        loss = clone(X).norm(p=2)
        clone.adapt(loss)
        new_loss = clone(X).norm(p=2)
        self.assertTrue(loss >= new_loss)
        new_loss.backward()
        for p in meta.parameters():
            self.assertTrue(hasattr(p, 'grad'))
            self.assertTrue(p.grad.norm(p=2).item() > 0.0)


if __name__ == '__main__':
    unittest.main()
