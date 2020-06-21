#!/usr/bin/env python3

import unittest
import random
import numpy as np

import torch
from torch import nn, optim
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader

import learn2learn as l2l


def pairwise_distances_logits(a, b):
    n = a.shape[0]
    m = b.shape[0]
    logits = -((a.unsqueeze(1).expand(n, m, -1) -
                b.unsqueeze(0).expand(n, m, -1))**2).sum(dim=2)
    return logits


def conv_block(in_channels, out_channels):
    bn = nn.BatchNorm2d(out_channels)
    nn.init.uniform_(bn.weight)
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        bn,
        nn.ReLU(),
        nn.MaxPool2d(2)
    )


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


class Convnet(nn.Module):

    # TODO: Is this architecture better than the one we have
    # in l2l.vision.models.ConvBase ?

    def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(x_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
        )
        self.out_channels = 1600

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)


def fast_adapt(model, batch, ways, shot, query_num, metric=None, device=None):
    if metric is None:
        metric = pairwise_distances_logits
    if device is None:
        device = model.device()
    data, labels = batch
    data = data.to(device)
    labels = labels.to(device)
    n_items = shot * ways

    # Sort data samples by labels
    # TODO: Can this be replaced by ConsecutiveLabels ?
    sort = torch.sort(labels)
    data = data.squeeze(0)[sort.indices].squeeze(0)
    labels = labels.squeeze(0)[sort.indices].squeeze(0)

    # Compute support and query embeddings
    embeddings = model(data)
    support_indices = np.zeros(data.size(0), dtype=bool)
    selection = np.arange(ways) * (shot + query_num)
    for offset in range(shot):
        support_indices[selection + offset] = True
    query_indices = torch.from_numpy(~support_indices)
    support_indices = torch.from_numpy(support_indices)
    support = embeddings[support_indices]
    support = support.reshape(ways, shot, -1).mean(dim=1)
    query = embeddings[query_indices]
    labels = labels[query_indices].long()

    logits = pairwise_distances_logits(query, support)
    loss = F.cross_entropy(logits, labels)
    acc = accuracy(logits, labels)
    return loss, acc


class Object:
    pass


def main(num_iterations=250):
    args = Object()
    setattr(args, 'max_epoch', num_iterations)
    setattr(args, 'shot', 1)
    setattr(args, 'test_way', 5)
    setattr(args, 'test_shot', 1)
    setattr(args, 'test_query', 30)
    setattr(args, 'train_query', 15)
    setattr(args, 'train_way', 30)
    setattr(args, 'gpu', 0)

    device = torch.device('cpu')
    if torch.cuda.device_count():
        torch.cuda.manual_seed(43)
        device = torch.device('cuda')

    model = Convnet()
    model.to(device)

    path_data = '~/data'
    train_dataset = l2l.vision.datasets.MiniImagenet(
        root=path_data, mode='train')
    valid_dataset = l2l.vision.datasets.MiniImagenet(
        root=path_data, mode='validation')
    test_dataset = l2l.vision.datasets.MiniImagenet(
        root=path_data, mode='test')

    train_dataset = l2l.data.MetaDataset(train_dataset)
    train_transforms = [
        l2l.data.transforms.NWays(train_dataset, args.train_way),
        l2l.data.transforms.KShots(train_dataset, args.train_query + args.shot),
        l2l.data.transforms.LoadData(train_dataset),
        l2l.data.transforms.RemapLabels(train_dataset),
    ]
    train_tasks = l2l.data.TaskDataset(train_dataset, task_transforms=train_transforms)
#    train_loader = DataLoader(train_tasks, pin_memory=True, shuffle=True)

    valid_dataset = l2l.data.MetaDataset(valid_dataset)
    valid_transforms = [
        l2l.data.transforms.NWays(valid_dataset, args.test_way),
        l2l.data.transforms.KShots(valid_dataset, args.test_query + args.test_shot),
        l2l.data.transforms.LoadData(valid_dataset),
        l2l.data.transforms.RemapLabels(valid_dataset),
    ]
    valid_tasks = l2l.data.TaskDataset(valid_dataset,
                                       task_transforms=valid_transforms,
                                       num_tasks=200)
#    valid_loader = DataLoader(valid_tasks, pin_memory=True, shuffle=True)

    test_dataset = l2l.data.MetaDataset(test_dataset)
    test_transforms = [
        l2l.data.transforms.NWays(test_dataset, args.test_way),
        l2l.data.transforms.KShots(test_dataset, args.test_query + args.test_shot),
        l2l.data.transforms.LoadData(test_dataset),
        l2l.data.transforms.RemapLabels(test_dataset),
    ]
    test_tasks = l2l.data.TaskDataset(test_dataset,
                                      task_transforms=test_transforms,
                                      num_tasks=200)
#    test_loader = DataLoader(test_tasks, pin_memory=True, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=20, gamma=0.5)

    for epoch in range(1, args.max_epoch + 1):
        model.train()

        loss_ctr = 0
        n_loss = 0
        n_acc = 0

        for i in range(100):
#            batch = next(iter(train_loader))
            batch = train_tasks.sample()

            loss, acc = fast_adapt(model,
                                   batch,
                                   args.train_way,
                                   args.shot,
                                   args.train_query,
                                   metric=pairwise_distances_logits,
                                   device=device)

            loss_ctr += 1
            n_loss += loss.item()
            n_acc += acc

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        lr_scheduler.step()

        print('epoch {}, train, loss={:.4f} acc={:.4f}'.format(
            epoch, n_loss/loss_ctr, n_acc/loss_ctr))
        train_accuracy = n_acc / loss_ctr

        model.eval()

        loss_ctr = 0
        n_loss = 0
        n_acc = 0
        for i, batch in enumerate(valid_tasks):
            loss, acc = fast_adapt(model,
                                   batch,
                                   args.test_way,
                                   args.test_shot,
                                   args.test_query,
                                   metric=pairwise_distances_logits,
                                   device=device)

            loss_ctr += 1
            n_loss += loss.item()
            n_acc += acc

        print('epoch {}, val, loss={:.4f} acc={:.4f}'.format(
            epoch, n_loss/loss_ctr, n_acc/loss_ctr))
        valid_accuracy = n_acc / loss_ctr

    loss_ctr = 0
    n_acc = 0

    for i, batch in enumerate(test_tasks, 1):
        loss, acc = fast_adapt(model,
                               batch,
                               args.test_way,
                               args.test_shot,
                               args.test_query,
                               metric=pairwise_distances_logits,
                               device=device)
        loss_ctr += 1
        n_acc += acc
    print('batch {}: {:.2f}({:.2f})'.format(
        i, n_acc/loss_ctr * 100, acc * 100))
    test_accuracy = n_acc / loss_ctr
    return train_accuracy, valid_accuracy, test_accuracy


class ProtoNetMiniImageNetIntegrationTests(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_final_accuracy(self):
        train_acc, valid_acc, test_acc = main(num_iterations=1)
        self.assertTrue(train_acc > 0.03)
        self.assertTrue(valid_acc > 0.20)
        self.assertTrue(test_acc > 0.20)


if __name__ == '__main__':
    unittest.main()
