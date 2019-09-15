#!/usr/bin/env python3

import unittest

import random
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

import learn2learn as l2l


class Net(nn.Module):
    def __init__(self, ways=3):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, ways)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1)
    acc = (predictions == targets).sum().float()
    acc /= len(targets)
    return acc.item()


def compute_loss(task, device, learner, loss_func, batch=5):
    loss = 0.0
    acc = 0.0
    dataloader = DataLoader(task, batch_size=batch, shuffle=False, num_workers=0)
    for i, (x, y) in enumerate(dataloader):
        x, y = x.squeeze(dim=1).to(device), y.view(-1).to(device)
        output = learner(x)
        curr_loss = loss_func(output, y)
        acc += accuracy(output, y)
        loss += curr_loss / x.size(0)
    loss /= len(dataloader)
    return loss, acc


def main(lr=0.005, maml_lr=0.01, iterations=1000, ways=5, shots=1, tps=32, fas=5, device=torch.device("cpu"),
         download_location="/tmp/datasets"):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    transformations = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        lambda x: x.view(1, 1, 28, 28),
    ])

    mnist_train = l2l.data.MetaDataset(MNIST(download_location, train=True, download=True, transform=transformations))
    # mnist_test = MNIST(file_location, train=False, download=True, transform=transformations)

    train_gen = l2l.data.TaskGenerator(mnist_train, ways=ways, tasks=10000)
    # test_gen = l2l.data.TaskGenerator(mnist_test, ways=ways)

    model = Net(ways)
    model.to(device)
    meta_model = l2l.algorithms.MAML(model, lr=maml_lr)
    opt = optim.Adam(meta_model.parameters(), lr=lr)
    loss_func = nn.NLLLoss(reduction="sum")

    for iteration in range(iterations):
        iteration_error = 0.0
        iteration_acc = 0.0
        for _ in range(tps):
            learner = meta_model.clone()
            train_task = train_gen.sample()
            valid_task = train_gen.sample(task=train_task.sampled_task)

            # Fast Adaptation
            for step in range(fas):
                train_error, _ = compute_loss(train_task, device, learner, loss_func, batch=shots * ways)
                learner.adapt(train_error)

            # Compute validation loss
            valid_error, valid_acc = compute_loss(valid_task, device, learner, loss_func, batch=shots * ways)
            iteration_error += valid_error
            iteration_acc += valid_acc

        iteration_error /= tps
        iteration_acc /= tps

        # Take the meta-learning step
        opt.zero_grad()
        iteration_error.backward()
        opt.step()
    return iteration_acc


class MNISTIntegrationTests(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_final_accuracy(self):
        result = main(iterations=20)
        self.assertTrue(result > 0.2)


if __name__ == '__main__':
    unittest.main()
