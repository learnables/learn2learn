#!/usr/bin/env python3

import copy
import unittest
import torch
import learn2learn as l2l

NUM_INPUTS = 7
INPUT_SIZE = 10
HIDDEN_SIZE = 20
INNER_LR = 0.01
EPSILON = 1e-8


def close(x, y):
    return (x - y).norm(p=2) <= EPSILON


class TestMAMLAlgorithm(unittest.TestCase):

    def setUp(self):
        self.model = torch.nn.Sequential(
            torch.nn.Linear(INPUT_SIZE, HIDDEN_SIZE),
            torch.nn.ReLU(),
            torch.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            torch.nn.Sigmoid(),
            torch.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            torch.nn.Softmax(),
        )

        self.model.register_buffer('dummy_buf', torch.zeros(1, 2, 3, 4))

    def tearDown(self):
        pass

    def test_clone_module(self):
        for first_order in [False, True]:
            maml = l2l.algorithms.MAML(self.model,
                                       lr=INNER_LR,
                                       first_order=first_order)
            X = torch.randn(NUM_INPUTS, INPUT_SIZE)
            ref = self.model(X)
            for clone in [maml.clone(), maml.clone()]:
                out = clone(X)
                self.assertTrue(close(ref, out))

    def test_first_order_adaptation(self):
        self.model.zero_grad()
        maml = l2l.algorithms.MAML(self.model,
                                   lr=INNER_LR,
                                   first_order=True)
        X = torch.randn(NUM_INPUTS, INPUT_SIZE)
        clone = maml.clone()
        # Fast adapt the clone
        for step in range(3):
            out = clone(X)
            loss = out.norm()
            clone.adapt(loss)
        # Create a reference as copy of clone
        ref = copy.deepcopy(self.model)
        with torch.no_grad():
            for rp, cp in zip(ref.parameters(), clone.parameters()):
                rp.data.copy_(cp.data)
        ref(X).norm().backward()
        # Compute first-order gradients
        loss = clone(X).norm()
        loss.backward()
        for pm, pr in zip(self.model.parameters(), ref.parameters()):
            self.assertTrue(close(pm.grad, pr.grad))

    def test_graph_connection(self):
        maml = l2l.algorithms.MAML(self.model,
                                   lr=INNER_LR,
                                   first_order=False)
        X = torch.randn(NUM_INPUTS, INPUT_SIZE)
        ref = maml(X)
        clone = maml.clone()
        out = clone(X)
        out.norm(p=2).backward()
        for p in self.model.parameters():
            self.assertTrue(hasattr(p, 'grad'))
            self.assertTrue(p.grad.norm(p=2).item() > 0.0)

    def test_adaptation(self):
        maml = l2l.algorithms.MAML(self.model,
                                   lr=INNER_LR,
                                   first_order=False)
        X = torch.randn(NUM_INPUTS, INPUT_SIZE)
        clone = maml.clone()
        loss = clone(X).norm(p=2)
        clone.adapt(loss)
        new_loss = clone(X).norm(p=2)
        self.assertTrue(loss >= new_loss)
        new_loss.backward()
        for p in self.model.parameters():
            self.assertTrue(hasattr(p, 'grad'))
            self.assertTrue(p.grad.norm(p=2).item() > 0.0)

    def test_allow_unused(self):
        maml = l2l.algorithms.MAML(self.model,
                                   lr=INNER_LR,
                                   first_order=False,
                                   allow_unused=True)
        clone = maml.clone()
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
        for p in maml.parameters():
            self.assertTrue(hasattr(p, 'grad'))

    def test_allow_nograd(self):
        self.model[2].weight.requires_grad = False
        maml = l2l.algorithms.MAML(self.model,
                                   lr=INNER_LR,
                                   first_order=False,
                                   allow_unused=False,
                                   allow_nograd=False)
        clone = maml.clone()

        loss = sum([p.norm(p=2) for p in clone.parameters()])
        try:
            # Check that without allow_nograd, adaptation fails
            clone.adapt(loss)
            # Check that execution never gets here
            self.assertTrue(
                False,
                'adaptation successful despite requires_grad=False',
            )
        except:
            # Check that with allow_nograd, adaptation succeeds
            clone.adapt(loss, allow_nograd=True)
            loss = sum([p.norm(p=2) for p in clone.parameters()])
            loss.backward()
            self.assertTrue(self.model[2].weight.grad is None)
            for p in self.model.parameters():
                if p.requires_grad:
                    self.assertTrue(p.grad is not None)

        maml = l2l.algorithms.MAML(
            self.model,
            lr=INNER_LR,
            first_order=False,
            allow_nograd=True,
        )
        clone = maml.clone()
        loss = sum([p.norm(p=2) for p in clone.parameters()])
        # Check that without allow_nograd, adaptation succeeds thanks to init.
        orig_weight = self.model[2].weight.clone().detach()
        clone.adapt(loss)
        self.assertTrue(close(orig_weight, self.model[2].weight))

    def test_module_shared_params(self):

        class TestModule(torch.nn.Module):
            def __init__(self):
                super(TestModule, self).__init__()
                cnn = [
                    torch.nn.Conv2d(3, 32, 3, 2, 1),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(32, 32, 3, 2, 1),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(32, 32, 3, 2, 1),
                    torch.nn.ReLU(),
                ]
                self.seq = torch.nn.Sequential(*cnn)
                self.head = torch.nn.Sequential(*[
                    torch.nn.Conv2d(32, 32, 3, 2, 1),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(32, 100, 3, 2, 1)]
                )
                self.net = torch.nn.Sequential(self.seq, self.head)

            def forward(self, x):
                return self.net(x)

        module = TestModule()
        maml = l2l.algorithms.MAML(module, lr=0.1)
        clone = maml.clone()
        loss = sum(p.norm(p=2) for p in clone.parameters())
        clone.adapt(loss)
        loss = sum(p.norm(p=2) for p in clone.parameters())
        loss.backward()

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
            maml = l2l.algorithms.MAML(model, lr=0.0001)
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
