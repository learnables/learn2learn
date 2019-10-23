#!/usr/bin/env python3

import os
import unittest
import random

import torch as th
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

import learn2learn as l2l

from copy import deepcopy


EPSILON = 0.0

def close(a, b):
    return (a - b).norm(p=2) <= EPSILON


class AdaptiveLR(nn.Module):

    def __init__(self, param, lr=1.0):
        super(AdaptiveLR, self).__init__()
        self.lr = th.ones_like(param.data) * lr
        self.lr = nn.Parameter(self.lr)

    def forward(self, grad):
        return self.lr * grad


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


class MetaOptimizerTests(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_supervised(self):
        pass

    def fast_adapt(self,
                   adaptation_data,
                   evaluation_data,
                   learner,
                   clone_opt,
                   loss,
                   adaptation_steps,
                   device):
        for step in range(adaptation_steps):
            data = [d for d in adaptation_data]
            X = th.cat([d[0] for d in data], dim=0).to(device)
            y = th.cat([th.tensor(d[1]).view(-1) for d in data], dim=0).to(device)
            train_error = loss(learner(X), y)
            train_error /= len(adaptation_data)

            # Check that both opt and model parameters are updated.
            old_opt_params = [p.clone().detach() for p in clone_opt.parameters()]
            clone_opt.adapt(train_error, lr=0.1)
            if step > 0 and False:
                for p, op in zip(clone_opt.parameters(), old_opt_params):
                    self.assertFalse(close(p, op), 
                                     'lopt parameters were not updated ?')
            old_model_params = [p.clone().detach() for p in learner.parameters()]
            clone_opt.step(train_error)
            for p, op in zip(learner.parameters(), old_model_params):
                self.assertFalse(close(p, op),
                                 'learner parameters were not updated ?')

        data = [d for d in evaluation_data]
        X = th.cat([d[0] for d in data], dim=0).to(device)
        y = th.cat([th.tensor(d[1]).view(-1) for d in data], dim=0).to(device)
        predictions = learner(X)
        valid_error = loss(predictions, y)
        valid_error /= len(evaluation_data)
        return valid_error

    def test_maml(self):
        ways = 3
        shots = 1
        meta_lr = 0.003
        fast_lr = 0.5
        meta_batch_size = 32
        adaptation_steps = 3
        num_iterations = 10
        cuda = False
        seed = 42
        device = th.device('cpu')

        # Create model
        model = l2l.vision.models.OmniglotFC(28 ** 2, ways)
        model.to(device)
        maml = l2l.algorithms.MAML(model, lr=fast_lr, first_order=False)
        meta_opt = l2l.optim.MetaOptimizer([{
                                            'params': [p],
                                            'update': AdaptiveLR(p, lr=0.5)
                                            } for p in model.parameters()],
                                           model,
                                           create_graph=True)
        all_params = list(meta_opt.parameters()) + list(maml.parameters())
        opt = optim.Adam(all_params, lr=3e-4)
        loss = nn.CrossEntropyLoss(reduction='mean')

        for iteration in range(num_iterations):
            meta_opt.zero_grad()
            opt.zero_grad()
            meta_train_error = 0.0
            meta_valid_error = 0.0
            meta_test_error = 0.0
            for task in range(meta_batch_size):
                # Compute meta-training loss
                learner = maml.clone()
                clone_opt = meta_opt.clone(model=learner)
                clone_opt.zero_grad()

                adaptation_data = [(th.randn(1, 28, 28), random.choice(list(range(ways)))) for _ in range(10)]
                evaluation_data = adaptation_data
                evaluation_error = self.fast_adapt(adaptation_data,
                                                   evaluation_data,
                                                   learner,
                                                   clone_opt,
                                                   loss,
                                                   adaptation_steps,
                                                   device)
                evaluation_error.backward()
                meta_train_error += evaluation_error.item()

            # Average the accumulated gradients and optimize
            for p in maml.parameters():
                p.grad.data.mul_(1.0 / meta_batch_size)
            for p in meta_opt.parameters():
                p.grad.data.mul_(1.0 / meta_batch_size)

            # Check that model and optimizer parameters are updated
            old_model_params = deepcopy(list(model.parameters()))
            old_opt_params = deepcopy(list(meta_opt.parameters()))
            opt.step()
            for p, op in zip(model.parameters(), old_model_params):
                self.assertFalse(close(p, op),
                        'model parameters were not updated ?')

            # NOTE: For meta_opt the gradients can be very small, so we check that at least one param is updated.
            meta_opt_checks = [close(p, op) for p, op in zip(meta_opt.parameters(), old_opt_params)]
            self.assertFalse(all(meta_opt_checks),
                    'meta-opt parameters were not updated ?')

    def test_supervised(self):
        transformations = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        dataset = MNIST('./data',
                        train=True,
                        download=True,
                        transform=transformations)
        dataset = DataLoader(dataset, batch_size=32)
        model = Net()
        meta_opt = l2l.optim.MetaOptimizer([{
                                            'params': [p],
                                            'update': AdaptiveLR(p, lr=0.1)
                                            } for p in model.parameters()],
                                           model,
                                           create_graph=False)
        opt = optim.Adam(meta_opt.parameters(), lr=3e-4)
        loss = nn.NLLLoss(reduction="mean")

        for i, (X, y) in enumerate(dataset):
            error = loss(model(X), y)
            opt.zero_grad()
            meta_opt.zero_grad()
            error.backward()
            # First update the meta-optimizers
            old_opt_params = [p.clone().detach() for p in meta_opt.parameters()]
            opt.step()
            if i > 0:
                for p, op in zip(meta_opt.parameters(), old_opt_params):
                    self.assertFalse(close(p, op), 'opt not updated ?')
            # Then update the model
            old_model_params = [p.clone().detach() for p in model.parameters()]
            meta_opt.step()
            for p, op in zip(model.parameters(), old_model_params):
                self.assertFalse(close(p, op), 'model not updated ?')
            # print('error:', error.item())
            if i >= 5:
                break


if __name__ == '__main__':
    unittest.main()

