#!/usr/bin/env python3

"""
Meta-Learning with SimpleShot.

Training: classical multiclass supervised learning.
Testing: Use centered and L2-normalized features.
"""

import os
import random

import numpy as np
import torch
from torch import nn

import learn2learn as l2l

from learn2learn.data.transforms import FusedNWaysKShots, LoadData, RemapLabels, ConsecutiveLabels
from torchvision.transforms import ToTensor


class Lambda(torch.nn.Module):

    def __init__(self, lmb):
        super(Lambda, self).__init__()
        self.lmb = lmb

    def forward(self, x):
        return self.lmb(x)


class CifarFSCNN(torch.nn.Module):

    def __init__(self, output_size):
        super(CifarFSCNN, self).__init__()
        self.features = torch.nn.Sequential(
            l2l.vision.models.ConvBase(output_size=64,
                                       hidden=64,
                                       channels=3,
                                       layers=4,
                                       max_pool=True,
                                       max_pool_factor=1.0,
                                       ),
            Lambda(lambda x: x.view(-1, 256))
        )
        self.head = torch.nn.Linear(256, output_size, bias=True)
        l2l.vision.models.maml_init_(self.head)

    def forward(self, x):
        x = self.features(x)
        x = self.head(x)
        return x


def pairwise_distances_logits(a, b):
    n = a.shape[0]
    m = b.shape[0]
    logits = -((a.unsqueeze(1).expand(n, m, -1) -
                b.unsqueeze(0).expand(n, m, -1))**2).sum(dim=2)
    return logits


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


def fast_adapt(batch, model, x_mean, loss, query_num, shots, ways, metric=None, device=None):
    if metric is None:
        metric = pairwise_distances_logits
    if device is None:
        device = model.device()
    data, labels = batch
    data = data.to(device)
    labels = labels.to(device)
    n_items = shots * ways

    # Sort data samples by labels
    sort = torch.sort(labels)
    data = data.squeeze(0)[sort.indices].squeeze(0)
    labels = labels.squeeze(0)[sort.indices].squeeze(0)

    # Compute support and query embeddings
    embeddings = model(data) - x_mean
    embeddings = embeddings / embeddings.norm(p=2, dim=1, keepdim=True)
    support_indices = np.zeros(data.size(0), dtype=bool)
    selection = np.arange(ways) * (shots + query_num)
    for offset in range(shots):
        support_indices[selection + offset] = True
    query_indices = torch.from_numpy(~support_indices)
    support_indices = torch.from_numpy(support_indices)
    support = embeddings[support_indices]
    support = support.reshape(ways, shots, -1).mean(dim=1)
    query = embeddings[query_indices]
    labels = labels[query_indices].long()

    logits = pairwise_distances_logits(query, support)
    loss = loss(logits, labels)
    acc = accuracy(logits, labels)
    return loss, acc


def main(
        test_ways=5,
        test_shots=5,
        test_queries=5,
        lr=0.003,
        meta_bsz=64,
        iters=200,
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
    train_dataset = l2l.vision.datasets.CIFARFS(root='~/data',
                                                mode='train',
                                                transform=ToTensor())
    valid_dataset = l2l.vision.datasets.CIFARFS(root='~/data',
                                                mode='validation',
                                                transform=ToTensor())
    test_dataset = l2l.vision.datasets.CIFARFS(root='~/data',
                                               mode='test',
                                               transform=ToTensor())
    train_dataset = l2l.data.MetaDataset(train_dataset)
    valid_dataset = l2l.data.MetaDataset(valid_dataset)
    test_dataset = l2l.data.MetaDataset(test_dataset)

    train_transforms = [
        FusedNWaysKShots(train_dataset, n=test_ways, k=test_queries+test_shots),
        LoadData(train_dataset),
        RemapLabels(train_dataset),
    ]
    train_tasks = l2l.data.TaskDataset(train_dataset, task_transforms=train_transforms)

    valid_transforms = [
        FusedNWaysKShots(valid_dataset, n=test_ways, k=test_queries+test_shots),
        LoadData(valid_dataset),
        RemapLabels(valid_dataset),
    ]
    valid_tasks = l2l.data.TaskDataset(valid_dataset,
                                       task_transforms=valid_transforms,
                                       num_tasks=200)

    test_transforms = [
        FusedNWaysKShots(test_dataset, n=test_ways, k=test_queries+test_shots),
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
    model = CifarFSCNN(output_size=64)
    model.to(device)
    features = model.features

    # Setup optimization
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.001)
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
