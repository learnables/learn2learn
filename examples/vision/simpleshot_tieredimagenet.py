#!/usr/bin/env python3

"""
Meta-Learning with SimpleShot.

Training: classical supervised learning.
Testing: Use centered and L2-normalized features.
"""

import os
import random

import numpy as np
import torch
from torch import nn

import learn2learn as l2l

from statistics import mean
from copy import deepcopy
from learn2learn.data.transforms import NWays, KShots, LoadData, RemapLabels, ConsecutiveLabels


class Lambda(nn.Module):

    def __init__(self, fn):
        super(Lambda, self).__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


def pairwise_distances_logits(a, b):
    n = a.shape[0]
    m = b.shape[0]
    logits = -((a.unsqueeze(1).expand(n, m, -1) -
                b.unsqueeze(0).expand(n, m, -1))**2).sum(dim=2)
    return logits


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


def fast_adapt(batch, model, x_mean, loss, query_num, shots, ways, num_clusters=1, metric=None, device=None):
    if metric is None:
        metric = pairwise_distances_logits
    if device is None:
        device = model.device()
    data, labels = batch
    data = data.to(device)
    labels = labels.to(device)
    n_items = num_clusters * shots * ways

    # Sort data samples by labels
    sort = torch.sort(labels)
    data = data.squeeze(0)[sort.indices].squeeze(0)
    labels = labels.squeeze(0)[sort.indices].squeeze(0)

    # Compute support and query embeddings
    embeddings = model(data)
    embeddings = embeddings / embeddings.norm(p=2, dim=1, keepdim=True)
    support_indices = torch.zeros(data.size(0)).byte()
    selection = torch.arange(ways) * (shots + query_num)
    for offset in range(shots):
        support_indices[selection + offset] = 1
    support = embeddings[support_indices]
    support = support.reshape(ways, shots, -1).mean(dim=1)
    query = embeddings[1 - support_indices]
    labels = labels[1 - support_indices].long()

    logits = pairwise_distances_logits(query, support)
    loss = loss(logits, labels)
    acc = accuracy(logits, labels)
    return loss, acc


def main(
        test_ways=10,
        test_shots=1,
        test_queries=5,
        lr=0.001,
        meta_bsz=32,
        iters=250,
        cuda=1,
        seed=42,
):

    cuda = bool(cuda)

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
        NWays(train_dataset, test_ways),
        KShots(train_dataset, test_queries + test_shots),
        LoadData(train_dataset),
        RemapLabels(train_dataset),
    ]
    train_tasks = l2l.data.TaskDataset(train_dataset, task_transforms=train_transforms)

    valid_transforms = [
        NWays(valid_dataset, test_ways),
        KShots(valid_dataset, test_queries + test_shots),
        LoadData(valid_dataset),
        RemapLabels(valid_dataset),
    ]
    valid_tasks = l2l.data.TaskDataset(valid_dataset,
                                       task_transforms=valid_transforms,
                                       num_tasks=200)

    test_transforms = [
        NWays(test_dataset, test_ways),
        KShots(test_dataset, test_queries + test_shots),
        LoadData(test_dataset),
        RemapLabels(test_dataset),
    ]
    test_tasks = l2l.data.TaskDataset(test_dataset,
                                      task_transforms=test_transforms,
                                      num_tasks=2000)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=32,
                                               shuffle=True)

    # Create model
    model = l2l.vision.models.MiniImagenetCNN(64)
    model.to(device)
    features = torch.nn.Sequential(model.base, Lambda(lambda x: x.view(-1, 800)))

    # Setup optimization
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss(reduction='mean')

    for iteration in range(iters):
        meta_train_error = 0.0
        meta_train_accuracy = 0.0
        meta_valid_error = 0.0
        meta_valid_accuracy = 0.0
        meta_test_error = 0.0
        meta_test_accuracy = 0.0

        # Do one epoch of supervised learning
        model.train()
        print('\n')
        print('Iteration', iteration)
        for train_X, train_y in train_loader:
            train_X = train_X.to(device)
            train_y = train_y.to(device).long()
            optimizer.zero_grad()
            train_preds = model(train_X)
            evaluation_error = loss(train_preds, train_y)
            evaluation_accuracy = accuracy(train_preds, train_y)
            evaluation_error.backward()
            optimizer.step()

        # Compute few-shot validation and test metrics
        model.eval()
        with torch.no_grad():
            x_mean = sum([features(x.to(device)).mean(dim=0, keepdim=True) for x, y in train_loader]) / len(train_loader)
            for task in range(meta_bsz):
                # Compute meta-validation loss
                batch = train_tasks.sample()
                evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                                   features,
                                                                   x_mean,
                                                                   loss,
                                                                   test_queries,
                                                                   test_shots,
                                                                   test_ways,
                                                                   num_clusters=1,
                                                                   device=device)
                meta_train_error += evaluation_error.item()
                meta_train_accuracy += evaluation_accuracy.item()


                # Compute meta-validation loss
                batch = valid_tasks.sample()
                evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                                   features,
                                                                   x_mean,
                                                                   loss,
                                                                   test_queries,
                                                                   test_shots,
                                                                   test_ways,
                                                                   num_clusters=1,
                                                                   device=device)
                meta_valid_error += evaluation_error.item()
                meta_valid_accuracy += evaluation_accuracy.item()

                # Compute meta-testing loss
                batch = test_tasks.sample()
                evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                                   features,
                                                                   x_mean,
                                                                   loss,
                                                                   test_queries,
                                                                   test_shots,
                                                                   test_ways,
                                                                   num_clusters=1,
                                                                   device=device)
                meta_test_error += evaluation_error.item()
                meta_test_accuracy += evaluation_accuracy.item()

        # Print some metrics
        print('Meta Train Error', meta_train_error / meta_bsz)
        print('Meta Train Accuracy', meta_train_accuracy / meta_bsz)
        print('Meta Valid Error', meta_valid_error / meta_bsz)
        print('Meta Valid Accuracy', meta_valid_accuracy / meta_bsz)
        print('Meta Test Error', meta_test_error / meta_bsz)
        print('Meta Test Accuracy', meta_test_accuracy / meta_bsz)
    

if __name__ == '__main__':
    main()
