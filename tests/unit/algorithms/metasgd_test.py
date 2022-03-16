#!/usr/bin/env python3

import unittest

import torch

import learn2learn as l2l

NUM_INPUTS = 7
INPUT_SIZE = 10
HIDDEN_SIZE = 20
INNER_LR = 0.1
EPSILON = 1e-8


def close(x, y):
    return (x - y).norm(p=2) <= EPSILON


class TestMetaSGDAlgorithm(unittest.TestCase):

    def setUp(self):
        self.model = torch.nn.Sequential(torch.nn.Linear(INPUT_SIZE, HIDDEN_SIZE),
                                         torch.nn.ReLU(),
                                         torch.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
                                         torch.nn.Sigmoid(),
                                         torch.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
                                         torch.nn.Softmax())
        self.model.register_buffer('dummy_buf', torch.zeros(1, 2, 3, 4))

    def tearDown(self):
        pass

    def test_clone_module(self):
        for first_order in [False, True]:
            meta = l2l.algorithms.MetaSGD(self.model,
                                          lr=INNER_LR,
                                          first_order=first_order)
            X = torch.randn(NUM_INPUTS, INPUT_SIZE)
            ref = self.model(X)
            for clone in [meta.clone(), meta.clone()]:
                out = clone(X)
                self.assertTrue(close(ref, out))

    def test_graph_connection(self):
        meta = l2l.algorithms.MetaSGD(self.model,
                                      lr=INNER_LR,
                                      first_order=False)
        X = torch.randn(NUM_INPUTS, INPUT_SIZE)
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
        X = torch.randn(NUM_INPUTS, INPUT_SIZE)
        clone = meta.clone()
        loss = clone(X).norm(p=2)
        clone.adapt(loss)
        new_loss = clone(X).norm(p=2)
        self.assertTrue(loss >= new_loss)
        new_loss.backward()
        for p in meta.parameters():
            self.assertTrue(hasattr(p, 'grad'))
            self.assertTrue(p.grad.norm(p=2).item() > 0.0)

    def test_memory_consumption(self):

        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            def get_memory():
                return torch.cuda.memory_allocated(0)

            BSZ = 1024
            INPUT_SIZE = 128
            N_STEPS = 5
            N_EVAL = 5

            device = torch.device("cuda")
            model = torch.nn.Sequential(*[
                torch.nn.Linear(INPUT_SIZE, INPUT_SIZE) for _ in range(10)
            ])
            maml = l2l.algorithms.MetaSGD(model, lr=0.0001)
            maml.to(device)

            memory_usages = []

            for evaluation in range(N_EVAL):
                learner = maml.clone()
                X = torch.randn(BSZ, INPUT_SIZE, device=device)
                for step in range(N_STEPS):
                    learner.adapt(torch.norm(learner(X)))
                memory_usages.append(get_memory())

            for i in range(1, len(memory_usages)):
                self.assertTrue(memory_usages[0] == memory_usages[i])


if __name__ == '__main__':
    unittest.main()
