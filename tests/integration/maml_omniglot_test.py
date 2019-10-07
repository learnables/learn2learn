#!/usr/bin/env python3

import unittest
import random
import numpy as np

import torch
from torch import nn, optim
from torchvision import transforms

import learn2learn as l2l

from copy import deepcopy
from PIL.Image import LANCZOS


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


def fast_adapt(adaptation_data, evaluation_data, learner, loss, adaptation_steps, device):
    for step in range(adaptation_steps):
        data = [d for d in adaptation_data]
        X = torch.cat([d[0] for d in data], dim=0).to(device)
        y = torch.cat([torch.tensor(d[1]).view(-1) for d in data], dim=0).to(device)
        train_error = loss(learner(X), y)
        train_error /= len(adaptation_data)
        learner.adapt(train_error)
    data = [d for d in evaluation_data]
    X = torch.cat([d[0] for d in data], dim=0).to(device)
    y = torch.cat([torch.tensor(d[1]).view(-1) for d in data], dim=0).to(device)
    predictions = learner(X)
    valid_error = loss(predictions, y)
    valid_error /= len(evaluation_data)
    valid_accuracy = accuracy(predictions, y)
    return valid_error, valid_accuracy

def main(
        ways=5,
        shots=1,
        meta_lr=0.003,
        fast_lr=0.5,
        meta_batch_size=32,
        adaptation_steps=1,
        num_iterations=60000,
        cuda=False,
        seed=42,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cpu')
    if cuda:
        torch.cuda.manual_seed(seed)
        device = torch.device('cuda')

    omniglot = l2l.vision.datasets.FullOmniglot(root='./data',
                                                transform=transforms.Compose([
                                                    l2l.vision.transforms.RandomDiscreteRotation(
                                                        [0.0, 90.0, 180.0, 270.0]),
                                                    transforms.Resize(28, interpolation=LANCZOS),
                                                    transforms.ToTensor(),
                                                    lambda x: 1.0 - x,
                                                ]),
                                                download=True)
    omniglot = l2l.data.MetaDataset(omniglot)
    classes = list(range(1623))
    random.shuffle(classes)
    train_generator = l2l.data.TaskGenerator(dataset=omniglot,
                                             ways=ways,
                                             classes=classes[:1100],
                                             tasks=20000)
    valid_generator = l2l.data.TaskGenerator(dataset=omniglot,
                                             ways=ways,
                                             classes=classes[1100:1200],
                                             tasks=1024)
    test_generator = l2l.data.TaskGenerator(dataset=omniglot,
                                            ways=ways,
                                            classes=classes[1200:],
                                            tasks=1024)

    # Create model
    model = l2l.vision.models.OmniglotFC(28 ** 2, ways)
    model.to(device)
    maml = l2l.algorithms.MAML(model, lr=fast_lr, first_order=False)
    opt = optim.Adam(maml.parameters(), meta_lr)
    loss = nn.CrossEntropyLoss(size_average=True, reduction='mean')

    for iteration in range(num_iterations):
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
            adaptation_data = train_generator.sample(shots=shots)
            evaluation_data = train_generator.sample(shots=shots,
                                                     task=adaptation_data.sampled_task)
            evaluation_error, evaluation_accuracy = fast_adapt(adaptation_data,
                                                               evaluation_data,
                                                               learner,
                                                               loss,
                                                               adaptation_steps,
                                                               device)
            evaluation_error.backward()
            meta_train_error += evaluation_error.item()
            meta_train_accuracy += evaluation_accuracy.item()

            # Compute meta-validation loss
            learner = deepcopy(maml)
            adaptation_data = valid_generator.sample(shots=shots)
            evaluation_data = valid_generator.sample(shots=shots,
                                                     task=adaptation_data.sampled_task)
            evaluation_error, evaluation_accuracy = fast_adapt(adaptation_data,
                                                               evaluation_data,
                                                               learner,
                                                               loss,
                                                               adaptation_steps,
                                                               device)
            meta_valid_error += evaluation_error.item()
            meta_valid_accuracy += evaluation_accuracy.item()

            # Compute meta-testing loss
            learner = deepcopy(maml)
            adaptation_data = test_generator.sample(shots=shots)
            evaluation_data = test_generator.sample(shots=shots,
                                                    task=adaptation_data.sampled_task)
            evaluation_error, evaluation_accuracy = fast_adapt(adaptation_data,
                                                               evaluation_data,
                                                               learner,
                                                               loss,
                                                               adaptation_steps,
                                                               device)
            meta_test_error += evaluation_error.item()
            meta_test_accuracy += evaluation_accuracy.item()

        # Print some metrics
        # print('\n')
        # print('Iteration', iteration)
        # print('Meta Train Error', meta_train_error / meta_batch_size)
        # print('Meta Train Accuracy', meta_train_accuracy / meta_batch_size)
        # print('Meta Valid Error', meta_valid_error / meta_batch_size)
        # print('Meta Valid Accuracy', meta_valid_accuracy / meta_batch_size)
        # print('Meta Test Error', meta_test_error / meta_batch_size)
        # print('Meta Test Accuracy', meta_test_accuracy / meta_batch_size)


        # Average the accumulated gradients and optimize
        for p in maml.parameters():
            p.grad.data.mul_(1.0 / meta_batch_size)
        opt.step()
    return meta_train_accuracy, meta_valid_accuracy, meta_test_accuracy


class MAMLOmniglotIntegrationTests(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_final_accuracy(self):
        train_acc, valid_acc, test_acc = main(num_iterations=35)
        self.assertTrue(train_acc > 0.5)
        self.assertTrue(valid_acc > 0.5)
        self.assertTrue(test_acc > 0.5)


if __name__ == '__main__':
    unittest.main()
