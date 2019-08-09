#!/usr/bin/env python3

import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from tqdm import tqdm

import learn2learn as l2l

WAYS = 3
SHOTS = 5
TASKS_PER_STEPS = 32


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


def inner_training_loop(task, device, learner, loss_func):
    loss = 0
    for i, (X, y) in enumerate(torch.utils.data.DataLoader(
            task, batch_size=15, shuffle=True, num_workers=0)):
        X, y = X.squeeze(dim=1).to(device), torch.tensor(y).view(-1).to(device)
        output = learner(X)
        curr_loss = loss_func(output, y)
        loss += curr_loss
        loss /= len(task)
    return loss


def main(file_location="/tmp/mnist"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mnist_train = MNIST(file_location, train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,)),
                            lambda x: x.view(1, 1, 28, 28),
                        ])
                        )
    mnist_test = MNIST(file_location, train=False, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,)),
                           lambda x: x.view(1, 1, 28, 28),
                       ])
                       )
    train_gen = l2l.data.TaskGenerator(mnist_train, ways=WAYS)
    test_gen = l2l.data.TaskGenerator(mnist_test, ways=WAYS)

    model = Net(WAYS)
    meta_model = l2l.MAML(model, lr=0.01)
    opt = optim.SGD(meta_model.parameters(), lr=0.001, momentum=0.9)
    loss_func = nn.NLLLoss(reduction="sum")

    for iteration in tqdm(range(1000)):
        iteration_error = 0.0
        for _ in range(TASKS_PER_STEPS):
            learner = meta_model.new()
            train_task = train_gen.sample(shots=SHOTS)
            valid_task = train_gen.sample(shots=SHOTS,
                                          classes_to_sample=train_task.sampled_classes)

            # Fast Adaptation
            for step in range(5):
                train_error = inner_training_loop(train_task, device, learner, loss_func)
                learner.adapt(train_error)

            # Compute validation loss
            valid_error = inner_training_loop(valid_task, device, learner, loss_func)
            iteration_error += valid_error

        iteration_error /= TASKS_PER_STEPS
        print('Valid error:', iteration_error.item())
        # Take the meta-learning step
        opt.zero_grad()
        iteration_error.backward()
        opt.step()


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()
