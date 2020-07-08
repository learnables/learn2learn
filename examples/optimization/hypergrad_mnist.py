#!/usr/bin/env python3

"""
File: hypergrad_mnist.py
Author: Seb Arnold - seba1511.net
Email: smr.arnold@gmail.com
Github: seba-1511
Description: Demonstation of the LearnableOptimizer to optimize a CNN on MNIST. 

While this example is inspired form the hypergradient literature, it differs
from Hypergradient:
    1. We do not use the analytical expression for the hypergradient, but
       instead rely on autograd to compute it for us.
    2. We learn a per-parameter learning rate rather than one shared across
       all parameters.

The network is inspired from the official MNIST example, in the PyTorch repo.
"""

import torch
from torch.nn import functional as F
import torchvision as tv
import learn2learn as l2l
import tqdm


def accuracy(predictions, targets):
    """Returns mean accuracy over a mini-batch"""
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, 3, 1)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = torch.nn.Dropout2d(0.25)
        self.dropout2 = torch.nn.Dropout2d(0.5)
        self.fc1 = torch.nn.Linear(9216, 128)
        self.fc2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class HypergradTransform(torch.nn.Module):
    """Hypergradient-style per-parameter learning rates"""

    def __init__(self, param, lr=0.01):
        super(HypergradTransform, self).__init__()
        self.lr = lr * torch.ones_like(param, requires_grad=True)
        self.lr = torch.nn.Parameter(self.lr)

    def forward(self, grad):
        return self.lr * grad


def main():
    torch.manual_seed(1234)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net()
    model.to(device)
    metaopt = l2l.optim.LearnableOptimizer(
        model=model,  # We pass the model, not its parameters
        transform=HypergradTransform,  # Any transform could work
        lr=0.1)
    metaopt.to(device)  # metaopt inherits from torch.nn.Module
    opt = torch.optim.Adam(metaopt.parameters(), lr=3e-4)
    loss = torch.nn.NLLLoss()

    kwargs = {'num_workers': 1,
              'pin_memory': True} if torch.cuda.is_available() else {}
    train_loader = torch.utils.data.DataLoader(
        tv.datasets.MNIST('~/data', train=True, download=True,
            transform=tv.transforms.Compose([
                tv.transforms.ToTensor(),
                tv.transforms.Normalize((0.1307,), (0.3081,))
                ])),
            batch_size=32, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        tv.datasets.MNIST('~/data', train=False, transform=tv.transforms.Compose([
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.1307,), (0.3081,))
            ])),
        batch_size=128, shuffle=False, **kwargs)

    for epoch in range(10):
        # Train for an epoch
        model.train()
        for X, y in tqdm.tqdm(train_loader, leave=False):
            X, y = X.to(device), y.to(device)
            metaopt.zero_grad()
            opt.zero_grad()
            err = loss(model(X), y)
            err.backward()
            opt.step()  # Update metaopt parameters
            metaopt.step()  # Update model parameters

        # Compute test error
        model.eval()
        test_error = 0.0
        test_accuracy = 0.0
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                preds = model(X)
                test_error += loss(preds, y)
                test_accuracy += accuracy(preds, y)
            test_error /= len(test_loader)
            test_accuracy /= len(test_loader)
        print('\nEpoch', epoch)
        print('Loss:', test_error.item())
        print('Accuracy:', test_accuracy.item())

    # Print the learned learning rates of the model
    print('The learning rates were:')
    for p in metaopt.parameters():
        print(p)


if __name__ == '__main__':
    main()
