#!/usr/bin/env python3

import torch
from torch import optim, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

import learn2learn as l2l


class AdaptiveLR(nn.Module):

    def __init__(self, param, lr=1.0):
        super(AdaptiveLR, self).__init__()
        self.lr = torch.ones_like(param.data) * lr
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


def main():
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

    for X, y in dataset:
        error = loss(model(X), y)
        opt.zero_grad()
        meta_opt.zero_grad()
        error.backward()
        # First update the meta-optimizers
        opt.step()
        # Then update the model
        meta_opt.step()
        print('error:', error.item())


if __name__ == '__main__':
    main()
