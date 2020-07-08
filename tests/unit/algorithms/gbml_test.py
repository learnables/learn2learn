#!/usr/bin/env python3

import unittest
import torch
import learn2learn as l2l

NUM_INPUTS = 7
INPUT_SIZE = 10
HIDDEN_SIZE = 20
INNER_LR = 0.01
EPSILON = 1e-8


class LR(torch.nn.Module):

    def __init__(self, input_size, output_size):
        super(LR, self).__init__()
        self.lr = torch.ones(input_size)
        self.lr = torch.nn.Parameter(self.lr)

    def forward(self, grad):
        return self.lr * grad


def close(x, y):
    return (x - y).norm(p=2) <= EPSILON


class TestGBMLgorithm(unittest.TestCase):

    def setUp(self):
        self.model = torch.nn.Sequential(
            torch.nn.Linear(INPUT_SIZE, HIDDEN_SIZE),
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
            transform = l2l.optim.transforms.ModuleTransform(LR)
            gbml = l2l.algorithms.GBML(self.model,
                                       transform=transform,
                                       first_order=first_order,
                                       lr=INNER_LR)
            X = torch.randn(NUM_INPUTS, INPUT_SIZE)
            ref = self.model(X)
            for clone in [gbml.clone(), gbml.clone()]:
                out = clone(X)
                self.assertTrue(close(ref, out))

    def test_graph_connection(self):
        for adapt_transform in [False, True]:
            transform = l2l.optim.transforms.ModuleTransform(LR)
            gbml = l2l.algorithms.GBML(self.model,
                                       transform=transform,
                                       adapt_transform=adapt_transform,
                                       lr=INNER_LR)
            X = torch.randn(NUM_INPUTS, INPUT_SIZE)
            ref = gbml(X)
            clone = gbml.clone()
            out = clone(X)
            out.norm(p=2).backward()
            for p in self.model.parameters():
                self.assertTrue(hasattr(p, 'grad'))
                self.assertTrue(p.grad.norm(p=2).item() > 0.0)

    def test_adaptation(self):
        for adapt_transform in [False, True]:
            transform = l2l.optim.transforms.ModuleTransform(LR)
            gbml = l2l.algorithms.GBML(self.model,
                                       transform=transform,
                                       adapt_transform=adapt_transform,
                                       lr=INNER_LR)
            X = torch.randn(NUM_INPUTS, INPUT_SIZE)
            clone = gbml.clone()
            loss = clone(X).norm(p=2)
            clone.adapt(loss)
            new_loss = clone(X).norm(p=2)
            self.assertTrue(loss >= new_loss)
            new_loss.backward()
            for p in self.model.parameters():
                self.assertTrue(hasattr(p, 'grad'))
                self.assertTrue(p.grad.norm(p=2).item() > 0.0)

    def test_allow_unused(self):
        transform = l2l.optim.transforms.ModuleTransform(LR)
        gbml = l2l.algorithms.GBML(self.model,
                                   transform=transform,
                                   lr=INNER_LR,
                                   allow_unused=True)
        clone = gbml.clone()
        loss = 0.0
        for i, p in enumerate(clone.parameters()):
            if i % 2 == 0:
                loss += p.norm(p=2)
        clone.adapt(loss)
        loss = 0.0
        for i, p in enumerate(clone.parameters()):
            if i % 2 == 0:
                loss += p.norm(p=2)
        loss.backward()
        for p in gbml.parameters():
            self.assertTrue(hasattr(p, 'grad'))

    def test_allow_nograd(self):
        self.model[2].weight.requires_grad = False
        transform = l2l.optim.transforms.ModuleTransform(LR)
        gbml = l2l.algorithms.GBML(self.model,
                                   transform=transform,
                                   lr=INNER_LR,
                                   allow_unused=False,
                                   allow_nograd=False)
        clone = gbml.clone()

        loss = sum([p.norm(p=2) for p in clone.parameters()])
        try:
            # Check that without allow_nograd, adaptation fails
            clone.adapt(loss)
            self.assertTrue(False, 'adaptation successful despite requires_grad=False')  # Check that execution never gets here
        except:
            # Check that with allow_nograd, adaptation succeeds
            clone.adapt(loss, allow_nograd=True)
            loss = sum([p.norm(p=2) for p in clone.parameters()])
            loss.backward()
            self.assertTrue(self.model[2].weight.grad is None)
            for p in self.model.parameters():
                if p.requires_grad:
                    self.assertTrue(p.grad is not None)

        transform = l2l.optim.transforms.ModuleTransform(LR)
        gbml = l2l.algorithms.GBML(self.model,
                                   transform=transform,
                                   lr=INNER_LR,
                                   allow_nograd=True)
        clone = gbml.clone()
        loss = sum([p.norm(p=2) for p in clone.parameters()])
        # Check that without allow_nograd, adaptation succeeds thanks to init.
        orig_weight = self.model[2].weight.clone().detach()
        clone.adapt(loss)
        self.assertTrue(close(orig_weight, self.model[2].weight))


if __name__ == '__main__':
    unittest.main()
