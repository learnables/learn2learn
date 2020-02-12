#!/usr/bin/env python3

import random
import unittest
from unittest import TestCase

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from learn2learn.data import MetaDataset, TaskDataset
from learn2learn.data.transforms import LoadData


NUM_TASKS = 10
NUM_DATA = 128
X_SHAPE = 16
Y_SHAPE = 10
EPSILON = 1e-6
SUBSET_SIZE = 5
WORKERS = 4
META_BSZ = 16


def task_equal(A, B):
    diff = 0.0
    for a, b in zip(A, B):
        diff += (a-b).pow(2).sum().item()
    return diff < EPSILON

def random_subset(task_description):
    return random.choices(task_description, k=SUBSET_SIZE)


class TestTaskDataset(TestCase):

    def test_instanciation(self):
        data = torch.randn(NUM_DATA, X_SHAPE)
        labels = torch.randint(0, Y_SHAPE, (NUM_DATA, ))
        dataset = TensorDataset(data, labels)
        task_dataset = TaskDataset(dataset, task_transforms=[LoadData(dataset)], num_tasks=NUM_TASKS)
        self.assertEqual(len(task_dataset), NUM_TASKS)

    def test_task_caching(self):
        data = torch.randn(NUM_DATA, X_SHAPE)
        labels = torch.randint(0, Y_SHAPE, (NUM_DATA, ))
        dataset = TensorDataset(data, labels)
        task_dataset = TaskDataset(dataset, task_transforms=[LoadData(dataset)], num_tasks=NUM_TASKS)
        tasks = []
        for i, task in enumerate(task_dataset, 1):
            tasks.append(task)
        self.assertEqual(i, NUM_TASKS)
        for ref, task in zip(tasks, task_dataset):
            self.assertTrue(task_equal(ref, task))

        for i in range(NUM_TASKS):
            ref = tasks[i]
            task = task_dataset[i]
            self.assertTrue(task_equal(ref, task))

    def test_infinite_tasks(self):
        data = torch.randn(NUM_DATA, X_SHAPE)
        labels = torch.randint(0, Y_SHAPE, (NUM_DATA, ))
        dataset = TensorDataset(data, labels)
        task_dataset = TaskDataset(dataset, task_transforms=[LoadData(dataset), random_subset])
        self.assertEqual(len(task_dataset), 1)
        prev = task_dataset.sample()
        for i, task in enumerate(task_dataset):
            self.assertFalse(task_equal(prev, task))
            prev = task
            if i > 4:
                break

    def test_task_transforms(self):
        data = torch.randn(NUM_DATA, X_SHAPE)
        labels = torch.randint(0, Y_SHAPE, (NUM_DATA, ))
        dataset = TensorDataset(data, labels)
        task_dataset = TaskDataset(dataset,
                                   task_transforms=[LoadData(dataset), random_subset],
                                   num_tasks=NUM_TASKS)
        for task in task_dataset:
            # Tests transforms on the task_description
            self.assertEqual(len(task[0]), SUBSET_SIZE)
            self.assertEqual(len(task[1]), SUBSET_SIZE)

            # Tests transforms on the data
            self.assertEqual(task[0].size(1), X_SHAPE)
            self.assertLessEqual(task[1].max(), Y_SHAPE - 1)
            self.assertGreaterEqual(task[1].max(), 0)

    def test_dataloader(self):
        data = torch.randn(NUM_DATA, X_SHAPE)
        labels = torch.randint(0, Y_SHAPE, (NUM_DATA, ))
        dataset = TensorDataset(data, labels)
        task_dataset = TaskDataset(dataset,
                                   task_transforms=[LoadData(dataset), random_subset],
                                   num_tasks=NUM_TASKS)
        task_loader = DataLoader(task_dataset,
                                 shuffle=True,
                                 batch_size=META_BSZ,
                                 num_workers=WORKERS,
                                 drop_last=True)
        for task_batch in task_loader:
            self.assertEqual(task_batch[0].shape, (META_BSZ, X_SHAPE))
            self.assertEqual(task_batch[1].shape, (META_BSZ, 1))


if __name__ == '__main__':
    unittest.main()
