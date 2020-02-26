#!/usr/bin/env python3

import random

import os
import numpy as np
import torch
from torch import nn
from torch import optim

import learn2learn as l2l
from learn2learn.data.transforms import NWays, KShots, LoadData, RemapLabels, ConsecutiveLabels
import json
import datetime

now = datetime.datetime.now()

try:
    import wandb as _wandb
    _wandb.init(project="l2l")
except ImportError:
    _has_wandb = False
else:
    _has_wandb = True

config = dict(
    ways=5,
    shots=5,
    meta_lr=0.003,
    fast_lr=0.5,
    meta_batch_size=32,
    adaptation_steps=1,
    num_iterations=30000,
    seed=42,
)

logger = {
    'date': now.strftime("%d_%m_%Hh%M"),
    'model_ID': str(config['seed']) + '_' + str(np.random.randint(1, 9999)),
    'parameters': config,
    'train': {'loss_history': [], 'acc_history': []},
    'valid': {'loss_history': [], 'acc_history': []},
    'test': {'loss_history': [], 'acc_history': []}
}

model_path = logger['date'] + '_' + logger['model_ID']
os.mkdir(model_path)
with open(model_path + '/logger.json', 'w') as fp:
    json.dump(logger, fp)


def get_mini_imagenet(ways, shots):
    # Create Datasets
    train_dataset = l2l.vision.datasets.MiniImagenet(root='~/data', mode='train')
    valid_dataset = l2l.vision.datasets.MiniImagenet(root='~/data', mode='validation')
    test_dataset = l2l.vision.datasets.MiniImagenet(root='~/data', mode='test')
    train_dataset = l2l.data.MetaDataset(train_dataset)
    valid_dataset = l2l.data.MetaDataset(valid_dataset)
    test_dataset = l2l.data.MetaDataset(test_dataset)

    train_transforms = [
        NWays(train_dataset, ways),
        KShots(train_dataset, 2 * shots),
        LoadData(train_dataset),
        RemapLabels(train_dataset),
        ConsecutiveLabels(train_dataset),
    ]
    train_tasks = l2l.data.TaskDataset(train_dataset,
                                       task_transforms=train_transforms,
                                       num_tasks=20000)

    valid_transforms = [
        NWays(valid_dataset, ways),
        KShots(valid_dataset, 2 * shots),
        LoadData(valid_dataset),
        ConsecutiveLabels(train_dataset),
        RemapLabels(valid_dataset),
    ]
    valid_tasks = l2l.data.TaskDataset(valid_dataset,
                                       task_transforms=valid_transforms,
                                       num_tasks=600)

    test_transforms = [
        NWays(test_dataset, ways),
        KShots(test_dataset, 2 * shots),
        LoadData(test_dataset),
        RemapLabels(test_dataset),
        ConsecutiveLabels(train_dataset),
    ]
    test_tasks = l2l.data.TaskDataset(test_dataset,
                                      task_transforms=test_transforms,
                                      num_tasks=600)

    return train_tasks, valid_tasks, test_tasks


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


def fast_adapt(batch, learner, loss, adaptation_steps, shots, ways, device):
    data, labels = batch
    data, labels = data.to(device), labels.to(device)

    # Separate data into adaptation/evalutation sets
    adaptation_indices = np.zeros(data.size(0), dtype=bool)
    adaptation_indices[np.arange(shots * ways) * 2] = True
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
        ways=config['ways'],
        shots=config['shots'],
        meta_lr=config['meta_lr'],
        fast_lr=config['fast_lr'],
        meta_batch_size=config['meta_batch_size'],
        adaptation_steps=config['adaptation_steps'],
        num_iterations=config['num_iterations'],
        cuda=True,
        seed=config['seed'],
):


    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cpu')
    if cuda and torch.cuda.device_count():
        torch.cuda.manual_seed(seed)
        device = torch.device('cuda')

    train_tasks, valid_tasks, test_tasks = get_mini_imagenet(ways, shots)

    # Create model
    model = l2l.vision.models.MiniImagenetCNN(ways)
    model.to(device)
    maml = l2l.algorithms.MAML(model, lr=fast_lr, first_order=False)
    opt = optim.Adam(maml.parameters(), meta_lr)
    loss = nn.CrossEntropyLoss(reduction='mean')

    if _has_wandb:
        _wandb.watch(model)

    for iteration in range(num_iterations):
        opt.zero_grad()
        meta_train_error = 0.0
        meta_train_accuracy = 0.0
        meta_valid_error = 0.0
        meta_valid_accuracy = 0.0
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

        # Print some metrics
        meta_train_error = meta_train_error / meta_batch_size
        meta_train_accuracy = meta_train_accuracy / meta_batch_size
        meta_valid_error = meta_valid_error / meta_batch_size
        meta_valid_accuracy = meta_valid_accuracy / meta_batch_size

        logger['train']['acc_t'] = meta_train_accuracy
        logger['valid']['acc_t'] = meta_valid_accuracy

        logger['train']['acc_history'].append(meta_train_accuracy)
        logger['valid']['acc_history'].append(meta_valid_accuracy)

        if _has_wandb:
            _wandb.log({'train_acc': meta_train_accuracy, 'valid_acc': meta_valid_accuracy})

        print('\n')
        print('Iteration', iteration)
        print('Meta Train Accuracy', logger['train']['acc_t'])
        print('Meta Valid Accuracy', logger['valid']['acc_t'])

        # Average the accumulated gradients and optimize
        for p in maml.parameters():
            p.grad.data.mul_(1.0 / meta_batch_size)
        opt.step()

    meta_test_error = 0.0
    meta_test_accuracy = 0.0
    for task in range(meta_batch_size):
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
    meta_test_error = meta_test_error / meta_batch_size
    meta_test_accuracy = meta_test_accuracy / meta_batch_size

    print('Meta Test Error', meta_test_error)
    print('Meta Test Accuracy', meta_test_accuracy)

    logger['test']['accuracy'] = meta_test_accuracy

    with open(model_path + '/logger.json', 'w') as fp:
        json.dump(logger, fp)

    model.save(model_path + "/model.h5")
    if _has_wandb:
        model.save(os.path.join(_wandb.run.dir, "model.h5"))


if __name__ == '__main__':
    main()
