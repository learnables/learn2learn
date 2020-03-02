#!/usr/bin/env python3

"""
Re-implementation of Reptile with L2L.

Running as-is should replicate the mini-ImageNet 5-ways, 5-shots results.
"""

import random

import numpy as np
import torch
from torch import nn
from torch import optim
from torch import autograd

import learn2learn as l2l
from learn2learn.data.transforms import NWays, KShots, LoadData, RemapLabels, ConsecutiveLabels

from statistics import mean
from copy import deepcopy


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


def fast_adapt(batch,
               learner,
               fast_lr,
               loss,
               adapt_steps,
               batch_size,
               shots,
               ways,
               opt,
               device):
    """
    Only use the adaptation data to update parameters. (evaluation is only indicative.)
    """
    data, labels = batch
    data, labels = data.to(device), labels.to(device)

    # Separate data into adaptation/evalutation sets
    adaptation_indices = np.zeros(data.size(0), dtype=bool)
    adaptation_indices[np.arange(shots*ways) * 2] = True
    evaluation_indices = torch.from_numpy(~adaptation_indices)
    adaptation_indices = torch.from_numpy(adaptation_indices)
    adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
    evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]

    # Adaptation steps
    adaptation_data = [d for d in adaptation_data]
    for step in range(adapt_steps):
        data = random.sample(adaptation_data, batch_size)
        adapt_X = torch.cat([d[0].unsqueeze(0) for d in data], dim=0).to(device)
        adapt_y = torch.cat([torch.tensor(d[1]).view(-1) for d in data], dim=0).to(device)
        opt.zero_grad()
        error = loss(learner(adapt_X), adapt_y)
        error.backward()
        opt.step()

    predictions = learner(evaluation_data)
    valid_error = loss(predictions, evaluation_labels)
    valid_accuracy = accuracy(predictions, evaluation_labels)
    return valid_error, valid_accuracy


def main(
        ways=5,
        train_shots=15,
        test_shots=5,
        meta_lr=1.0,
        meta_mom=0.0,
        meta_bsz=5,
        fast_lr=0.001,
        train_bsz=10,
        test_bsz=15,
        train_adapt_steps=8,
        test_adapt_steps=50,
        num_iterations=100000,
        test_interval=100,
        adam=0,  # Use adam or sgd for fast-adapt
        meta_decay=1,  # Linearly decay the meta-lr or not
        cuda=1,
        seed=42,
):

    cuda = bool(cuda)
    use_adam = bool(adam)
    meta_decay = bool(meta_decay)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cpu')
    if cuda and torch.cuda.device_count():
        torch.cuda.manual_seed(seed)
        device = torch.device('cuda')

    # Create Datasets
    train_dataset = l2l.vision.datasets.MiniImagenet(root='~/data', mode='train')
    valid_dataset = l2l.vision.datasets.MiniImagenet(root='~/data', mode='validation')
    test_dataset = l2l.vision.datasets.MiniImagenet(root='~/data', mode='test')
    train_dataset = l2l.data.MetaDataset(train_dataset)
    valid_dataset = l2l.data.MetaDataset(valid_dataset)
    test_dataset = l2l.data.MetaDataset(test_dataset)

    train_transforms = [
        NWays(train_dataset, ways),
        KShots(train_dataset, 2*train_shots),
        LoadData(train_dataset),
        RemapLabels(train_dataset),
        ConsecutiveLabels(train_dataset),
    ]
    train_tasks = l2l.data.TaskDataset(train_dataset,
                                       task_transforms=train_transforms,
                                       num_tasks=20000)

    valid_transforms = [
        NWays(valid_dataset, ways),
        KShots(valid_dataset, 2*test_shots),
        LoadData(valid_dataset),
        ConsecutiveLabels(valid_dataset),
        RemapLabels(valid_dataset),
    ]
    valid_tasks = l2l.data.TaskDataset(valid_dataset,
                                       task_transforms=valid_transforms,
                                       num_tasks=600)

    test_transforms = [
        NWays(test_dataset, ways),
        KShots(test_dataset, 2*test_shots),
        LoadData(test_dataset),
        RemapLabels(test_dataset),
        ConsecutiveLabels(test_dataset),
    ]
    test_tasks = l2l.data.TaskDataset(test_dataset,
                                      task_transforms=test_transforms,
                                      num_tasks=600)


    # Create model
    model = l2l.vision.models.MiniImagenetCNN(ways)
    model.to(device)
    if use_adam:
        opt = optim.Adam(model.parameters(), meta_lr, betas=(meta_mom, 0.999))
    else:
        opt = optim.SGD(model.parameters(), lr=meta_lr, momentum=meta_mom)
    adapt_opt = optim.Adam(model.parameters(), lr=fast_lr, betas=(0, 0.999))
    adapt_opt_state = adapt_opt.state_dict()
    loss = nn.CrossEntropyLoss(reduction='mean')

    for iteration in range(num_iterations):
        # anneal meta-lr
        if meta_decay:
            frac_done = float(iteration) / num_iterations
            new_lr = frac_done * meta_lr + (1 - frac_done) * meta_lr
            for pg in opt.param_groups:
                pg['lr'] = new_lr

        # zero-grad the parameters
        for p in model.parameters():
            p.grad = torch.zeros_like(p.data)

        meta_train_error = 0.0
        meta_train_accuracy = 0.0
        meta_valid_error = 0.0
        meta_valid_accuracy = 0.0
        meta_test_error = 0.0
        meta_test_accuracy = 0.0
        for task in range(meta_bsz):
            # Compute meta-training loss
            learner = deepcopy(model)
            adapt_opt = optim.Adam(learner.parameters(),
                                   lr=fast_lr,
                                   betas=(0, 0.999))
            adapt_opt.load_state_dict(adapt_opt_state)
            batch = train_tasks.sample()
            evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                               learner,
                                                               fast_lr,
                                                               loss,
                                                               adapt_steps=train_adapt_steps,
                                                               batch_size=train_bsz,
                                                               opt=adapt_opt,
                                                               shots=train_shots,
                                                               ways=ways,
                                                               device=device)
            adapt_opt_state = adapt_opt.state_dict()
            for p, l in zip(model.parameters(), learner.parameters()):
                p.grad.data.add_(-1.0, l.data)

            meta_train_error += evaluation_error.item()
            meta_train_accuracy += evaluation_accuracy.item()


            if iteration % test_interval == 0:
                # Compute meta-validation loss
                learner = deepcopy(model)
                adapt_opt = optim.Adam(learner.parameters(),
                                       lr=fast_lr,
                                       betas=(0, 0.999))
                adapt_opt.load_state_dict(adapt_opt_state)
                batch = valid_tasks.sample()
                evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                                   learner,
                                                                   fast_lr,
                                                                   loss,
                                                                   adapt_steps=test_adapt_steps,
                                                                   batch_size=test_bsz,
                                                                   opt=adapt_opt,
                                                                   shots=test_shots,
                                                                   ways=ways,
                                                                   device=device)
                meta_valid_error += evaluation_error.item()
                meta_valid_accuracy += evaluation_accuracy.item()

                # Compute meta-testing loss
                learner = deepcopy(model)
                adapt_opt = optim.Adam(learner.parameters(),
                                       lr=fast_lr,
                                       betas=(0, 0.999))
                adapt_opt.load_state_dict(adapt_opt_state)
                batch = test_tasks.sample()
                evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                                   learner,
                                                                   fast_lr,
                                                                   loss,
                                                                   adapt_steps=test_adapt_steps,
                                                                   batch_size=test_bsz,
                                                                   opt=adapt_opt,
                                                                   shots=test_shots,
                                                                   ways=ways,
                                                                   device=device)
                meta_test_error += evaluation_error.item()
                meta_test_accuracy += evaluation_accuracy.item()

        # Print some metrics
        print('\n')
        print('Iteration', iteration)
        print('Meta Train Error', meta_train_error / meta_bsz)
        print('Meta Train Accuracy', meta_train_accuracy / meta_bsz)
        if iteration % test_interval == 0:
            print('Meta Valid Error', meta_valid_error / meta_bsz)
            print('Meta Valid Accuracy', meta_valid_accuracy / meta_bsz)
            print('Meta Test Error', meta_test_error / meta_bsz)
            print('Meta Test Accuracy', meta_test_accuracy / meta_bsz)

        # Average the accumulated gradients and optimize
        for p in model.parameters():
            p.grad.data.mul_(1.0 / meta_bsz).add_(p.data)
        opt.step()


if __name__ == '__main__':
    main()
