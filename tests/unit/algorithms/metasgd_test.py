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
    return (x - y).norm(p=2) <= EPSILON


class TestMetaSGDAlgorithm(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = th.nn.Sequential(th.nn.Linear(INPUT_SIZE, HIDDEN_SIZE),
                                     th.nn.ReLU(),
                                     th.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
                                     th.nn.Sigmoid(),
                                     th.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
                                     th.nn.Softmax())
        cls.model.register_buffer('dummy_buf', th.zeros(1, 2, 3, 4))

    def tearDown(self):
        pass

    def test_clone_module(self):
        for first_order in [False, True]:
            meta = l2l.algorithms.MetaSGD(self.model,
                                          lr=INNER_LR,
                                          first_order=first_order)
            X = th.randn(NUM_INPUTS, INPUT_SIZE)
            ref = self.model(X)
            for clone in [meta.clone(), meta.clone()]:
                out = clone(X)
                self.assertTrue(close(ref, out))

    def test_graph_connection(self):
        meta = l2l.algorithms.MetaSGD(self.model,
                                      lr=INNER_LR,
                                      first_order=False)
        X = th.randn(NUM_INPUTS, INPUT_SIZE)
        ref = meta(X)
        clone = meta.clone()
        out = clone(X)
        out.norm(p=2).backward()
        for p in self.model.parameters():
            self.assertTrue(hasattr(p, 'grad'))
            self.assertTrue(p.grad.norm(p=2).item() > 0.0)

    def test_meta_lr(self):
        meta = l2l.algorithms.MetaSGD(self.model,
                                      lr=INNER_LR,
                                      first_order=False)
        num_params = sum([p.numel() for p in self.model.parameters()])
        meta_params = sum([p.numel() for p in meta.parameters()])
        self.assertEqual(2 * num_params, meta_params)
        for lr in meta.lrs:
            self.assertTrue(close(lr, INNER_LR))

    def test_adaptation(self):
        meta = l2l.algorithms.MetaSGD(self.model,
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
