#!/usr/bin/env python3

import unittest
import random
import numpy as np

import torch
from torch import nn, optim
from torchvision import transforms

import learn2learn as l2l

from PIL.Image import LANCZOS


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


def fast_adapt(batch, learner, loss, adaptation_steps, shots, ways, device):
    data, labels = batch
    data, labels = data.to(device), labels.to(device)

    # Separate data into adaptation/evalutation sets
    adaptation_indices = np.zeros(data.size(0), dtype=bool)
    adaptation_indices[np.arange(shots*ways) * 2] = True
    evaluation_indices = torch.from_numpy(~adaptation_indices)
    adaptation_indices = torch.from_numpy(adaptation_indices)
    adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
    evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]

    # Adapt the model
    for step in range(adaptation_steps):
        train_error = loss(learner(adaptation_data), adaptation_labels)
        train_error /= len(adaptation_data)
        learner.adapt(train_error)

    # Evaluate the adapted model
    predictions = learner(evaluation_data)
    valid_error = loss(predictions, evaluation_labels)
    valid_error /= len(evaluation_data)
    valid_accuracy = accuracy(predictions, evaluation_labels)
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
                                                    transforms.Resize(28, interpolation=LANCZOS),
                                                    transforms.ToTensor(),
                                                    lambda x: 1.0 - x,
                                                ]),
                                                download=True)

    dataset = l2l.data.MetaDataset(omniglot)
    classes = list(range(1623))
    random.shuffle(classes)

    train_transforms = [
        l2l.data.transforms.FusedNWaysKShots(dataset,
                                             n=ways,
                                             k=2*shots,
                                             filter_labels=classes[:1100]),
        l2l.data.transforms.LoadData(dataset),
        l2l.data.transforms.RemapLabels(dataset),
        l2l.data.transforms.ConsecutiveLabels(dataset),
        l2l.vision.transforms.RandomClassRotation(dataset, [0.0, 90.0, 180.0, 270.0])
    ]
    train_tasks = l2l.data.TaskDataset(dataset,
                                       task_transforms=train_transforms,
                                       num_tasks=20000)

    valid_transforms = [
        l2l.data.transforms.FusedNWaysKShots(dataset,
                                             n=ways,
                                             k=2*shots,
                                             filter_labels=classes[1100:1200]),
        l2l.data.transforms.LoadData(dataset),
        l2l.data.transforms.RemapLabels(dataset),
        l2l.data.transforms.ConsecutiveLabels(dataset),
        l2l.vision.transforms.RandomClassRotation(dataset, [0.0, 90.0, 180.0, 270.0])
    ]
    valid_tasks = l2l.data.TaskDataset(dataset,
                                       task_transforms=valid_transforms,
                                       num_tasks=1024)

    test_transforms = [
        l2l.data.transforms.FusedNWaysKShots(dataset,
                                             n=ways,
                                             k=2*shots,
                                             filter_labels=classes[1200:]),
        l2l.data.transforms.LoadData(dataset),
        l2l.data.transforms.RemapLabels(dataset),
        l2l.data.transforms.ConsecutiveLabels(dataset),
        l2l.vision.transforms.RandomClassRotation(dataset, [0.0, 90.0, 180.0, 270.0])
    ]
    test_tasks = l2l.data.TaskDataset(dataset,
                                      task_transforms=test_transforms,
                                      num_tasks=1024)

    # Create model
    model = l2l.vision.models.OmniglotFC(28 ** 2, ways)
    model.to(device)
    maml = l2l.algorithms.MAML(model, lr=fast_lr, first_order=False)
    opt = optim.Adam(maml.parameters(), meta_lr)
    loss = nn.CrossEntropyLoss(reduction='mean')

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
            batch = train_tasks.sample()
            evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                               learner,
                                                               loss,
                                                               adaptation_steps,
                                                               shots,
                                                               ways,
                                                               device)
            evaluation_error.backward()
            meta_train_error += evaluation_error.item()
            meta_train_accuracy += evaluation_accuracy.item()

            # Compute meta-validation loss
            learner = maml.clone()
            batch = valid_tasks.sample()
            evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                               learner,
                                                               loss,
                                                               adaptation_steps,
                                                               shots,
                                                               ways,
                                                               device)
            meta_valid_error += evaluation_error.item()
            meta_valid_accuracy += evaluation_accuracy.item()

            # Compute meta-testing loss
            learner = maml.clone()
            batch = test_tasks.sample()
            evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                               learner,
                                                               loss,
                                                               adaptation_steps,
                                                               shots,
                                                               ways,
                                                               device)
            meta_test_error += evaluation_error.item()
            meta_test_accuracy += evaluation_accuracy.item()

        # Print some metrics
#        print('\n')
#        print('Iteration', iteration)
#        print('Meta Train Error', meta_train_error / meta_batch_size)
        print(iteration, 'Meta Train Accuracy', meta_train_accuracy / meta_batch_size)
#        print('Meta Valid Error', meta_valid_error / meta_batch_size)
#        print('Meta Valid Accuracy', meta_valid_accuracy / meta_batch_size)
#        print('Meta Test Error', meta_test_error / meta_batch_size)
#        print('Meta Test Accuracy', meta_test_accuracy / meta_batch_size)


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
        train_acc, valid_acc, test_acc = main(num_iterations=5)
        self.assertTrue(train_acc > 0.35)
        self.assertTrue(valid_acc > 0.35)
        self.assertTrue(test_acc > 0.35)


if __name__ == '__main__':
    unittest.main()
