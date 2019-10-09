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


class AdaptiveLR(nn.Module):

    def __init__(self, param, lr=1.0):
        super(AdaptiveLR, self).__init__()
        self.lr = th.ones_like(param.data) * lr
        self.lr = nn.Parameter(self.lr)

    def forward(self, grad):
        return self.lr * grad


def clone(meta_opt, model=None):
    # TODO: What if you only want to optimize a subset of the model's parameters ?
    if model is None:
        model = meta_opt.model
    if len(meta_opt.param_groups) == 1:
        updates = [l2l.clone_module(meta_opt.param_groups['update']) for _ in model.parameters()]
    else:
        updates = [l2l.clone_module(pg['update']) for pg in meta_opt.param_groups]

    for p in model.parameters():
        p.retain_grad()

    new_opt = l2l.optim.MetaOptimizer([{
                                        'params': [p],
                                        'update': u
                                        } for p, u in zip(model.parameters(),
                                                          updates)],
                                       model,
                                       create_graph=True)
    return new_opt






#    new_opt = deepcopy(meta_opt)
#    import pdb; pdb.set_trace()
#    old_model = new_opt.model
#    new_opt.model = model
#
#    # TODO: This is not gonna work for models that are not fully-optimized by a single opt.
#    print('clone')
#    new_opt._update_references(list(old_model.parameters()),
#                               [id(p) for p in meta_opt.model.parameters()])
#
#    # TODO: The following is not doing what it should: it breaks the graph, whereas we want to use clone_module.
#    for p in new_opt.parameters():
#        p.detach_()
#        p.retain_grad()
#        p.requires_grad = True
#    return new_opt


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



        # Compute gradients w.r.t opt
        opt_grads = autograd.grad(train_error, clone_opt.parameters(), allow_unused=True, create_graph=True)
        # Update opt via GD
        for p, g in zip(clone_opt.parameters(), opt_grads):
            p.grad = g
        for group in clone_opt.param_groups:
            update = group['update']
            l2l.algorithms.maml_update(update, lr=0.1)
        # Compute gradients w.r.t. model
        learner_grads = autograd.grad(train_error, learner.parameters(), create_graph=True)
        # Update learner via opt
        for p, g in zip(learner.parameters(), learner_grads):
            p.grad = g
        print('step')
        clone_opt.step()
#        learner.adapt(train_error)



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
        cuda=True,
        seed=42,
):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    device = th.device('cpu')
    if cuda:
        th.cuda.manual_seed(seed)
        device = th.device('cuda')

#    omniglot = torchvision.datasets.MNIST(root='./data',
#                                          transform=transforms.Compose([
#                                              l2l.vision.transforms.RandomDiscreteRotation(
#                                                  [0.0, 90.0, 180.0, 270.0]),
#                                              transforms.Resize(28, interpolation=LANCZOS),
#                                              transforms.ToTensor(),
#                                              lambda x: 1.0 - x,
#                                          ]),
#                                          download=True)
#    omniglot = l2l.data.MetaDataset(omniglot)
#    classes = list(range(10))
#    train_generator = l2l.data.TaskGenerator(dataset=omniglot,
#                                             ways=ways,
#                                             classes=classes[:6],
#                                             tasks=20000)

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
    opt = optim.Adam(meta_opt.parameters(), lr=3e-4)
    loss = nn.CrossEntropyLoss(size_average=True, reduction='mean')

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
            # TODO: Make sure that accumulating grads in p.grad does not affect meta-opt
            learner = maml.clone()
            learner.zero_grad()
            clone_opt = clone(meta_opt, learner)
            # TODO: Fix the following ?
            for p in clone_opt.parameters():
                if hasattr(p, 'grad') and p.grad is not None:
                    p.grad = th.zeros_like(p.data)
#            adaptation_data = train_generator.sample(shots=shots)
#            evaluation_data = train_generator.sample(shots=shots,
#                                                     task=adaptation_data.sampled_task)
            adaptation_data = [(th.randn(1, 28, 28), 2) for _ in range(10)]
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
        opt.step()


if __name__ == '__main__':
    main()


if __name__ == '__main__':
    main()
