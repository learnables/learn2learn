#!/usr/bin/env python3

import random

import numpy as np
import torch as th
from PIL.Image import LANCZOS

from torch import nn
from torch import optim
from torch import autograd
import torchvision
from torchvision import transforms

from copy import deepcopy

import learn2learn as l2l

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


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


def fast_adapt(adaptation_data, evaluation_data, learner, clone_opt, loss, adaptation_steps, device):
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
                assert not close(p, op), \
                        'lopt parameters were not updated ?'
        old_model_params = [p.clone().detach() for p in learner.parameters()]
        clone_opt.step(train_error)
        for p, op in zip(learner.parameters(), old_model_params):
            assert not close(p, op), \
                    'learner parameters were not updated ?'

    data = [d for d in evaluation_data]
    X = th.cat([d[0] for d in data], dim=0).to(device)
    y = th.cat([th.tensor(d[1]).view(-1) for d in data], dim=0).to(device)
    predictions = learner(X)
    valid_error = loss(predictions, y)
    valid_error /= len(evaluation_data)
    valid_accuracy = accuracy(predictions, y)
    return valid_error, valid_accuracy


def main(
        ways=3,
        shots=1,
        meta_lr=0.003,
        fast_lr=0.5,
        meta_batch_size=32,
        adaptation_steps=3,
        num_iterations=60000,
        cuda=False,
        seed=42,
):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    device = th.device('cpu')
    if cuda:
        th.cuda.manual_seed(seed)
        device = th.device('cuda')

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
        meta_train_accuracy = 0.0
        meta_valid_error = 0.0
        meta_valid_accuracy = 0.0
        meta_test_error = 0.0
        meta_test_accuracy = 0.0
        for task in range(meta_batch_size):
            # Compute meta-training loss
            learner = maml.clone()
            clone_opt = meta_opt.clone(model=learner)
            clone_opt.zero_grad()

            adaptation_data = [(th.randn(1, 28, 28), random.choice(list(range(ways)))) for _ in range(10)]
            evaluation_data = adaptation_data
            evaluation_error, evaluation_accuracy = fast_adapt(adaptation_data,
                                                               evaluation_data,
                                                               learner,
                                                               clone_opt,
                                                               loss,
                                                               adaptation_steps,
                                                               device)
            evaluation_error.backward()
            meta_train_error += evaluation_error.item()
            meta_train_accuracy += evaluation_accuracy.item()

        # Print some metrics
        print('\n')
        print('Iteration', iteration)
        print('Meta Train Error', meta_train_error / meta_batch_size)
        print('Meta Train Accuracy', meta_train_accuracy / meta_batch_size)

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
            assert not close(p, op), \
                    'model parameters were not updated ?'

        # NOTE: For meta_opt the gradients can be very small, so we check that at least one param is updated.
        meta_opt_checks = [close(p, op) for p, op in zip(meta_opt.parameters(), old_opt_params)]
        assert not all(meta_opt_checks), \
                'meta-opt parameters were not updated ?'


if __name__ == '__main__':
    main()
