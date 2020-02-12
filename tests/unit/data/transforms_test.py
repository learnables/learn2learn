#!/usr/bin/env python3

import random
import unittest
from unittest import TestCase

import numpy as np
import torch
from torch.utils.data import TensorDataset

from learn2learn.data import MetaDataset, TaskDataset
from learn2learn.data.transforms import NWays, KShots, LoadData, FilterLabels, RemapLabels

NUM_TASKS = 128
NUM_DATA = 512
X_SHAPE = 16
Y_SHAPE = 10
EPSILON = 1e-6
SUBSET_SIZE = 5
WORKERS = 4
META_BSZ = 16


class TestTransforms(TestCase):

    def test_n_ways(self):
        data = torch.randn(NUM_DATA, X_SHAPE)
        labels = torch.randint(0, Y_SHAPE, (NUM_DATA, ))
        dataset = MetaDataset(TensorDataset(data, labels))
        for ways in range(1, 10):
            task_dataset = TaskDataset(dataset, 
                                       task_transforms=[NWays(dataset, n=ways), LoadData(dataset)],
                                       num_tasks=NUM_TASKS)
            for task in task_dataset:
                bins = task[1].bincount()
                num_classes = len(bins) - (bins == 0).sum()
                self.assertEqual(num_classes, ways)

    def test_k_shots(self):
        data = torch.randn(NUM_DATA, X_SHAPE)
        labels = torch.randint(0, Y_SHAPE, (NUM_DATA, ))
        dataset = MetaDataset(TensorDataset(data, labels))
        for replacement in [False, True]:
            for shots in range(1, 10):
                task_dataset = TaskDataset(dataset, 
                                           task_transforms=[KShots(dataset, k=shots, replacement=replacement),
                                                            LoadData(dataset)],
                                           num_tasks=NUM_TASKS)
                for task in task_dataset:
                    bins = task[1].bincount()
                    correct = (bins == shots).sum()
                    self.assertEqual(correct, Y_SHAPE)

    def test_load_data(self):
        data = torch.randn(NUM_DATA, X_SHAPE)
        labels = torch.randint(0, Y_SHAPE, (NUM_DATA, ))
        dataset = MetaDataset(TensorDataset(data, labels))
        task_dataset = TaskDataset(dataset, 
                                   task_transforms=[LoadData(dataset)],
                                   num_tasks=NUM_TASKS)
        for task in task_dataset:
            self.assertTrue(isinstance(task[0], torch.Tensor))
            self.assertTrue(isinstance(task[1], torch.Tensor))

    def test_filter_labels(self):
        data = torch.randn(NUM_DATA, X_SHAPE)
        labels = torch.randint(0, Y_SHAPE, (NUM_DATA, ))
        chosen_labels = random.sample(list(range(Y_SHAPE)), k=Y_SHAPE//2)
        dataset = MetaDataset(TensorDataset(data, labels))
        task_dataset = TaskDataset(dataset, 
                                   task_transforms=[FilterLabels(dataset, chosen_labels), LoadData(dataset)],
                                   num_tasks=NUM_TASKS)
        for task in task_dataset:
            for label in task[1]:
                self.assertTrue(label in chosen_labels)

    def test_remap_labels(self):
        data = torch.randn(NUM_DATA, X_SHAPE)
        labels = torch.randint(0, Y_SHAPE, (NUM_DATA, ))
        dataset = MetaDataset(TensorDataset(data, labels))
        for ways in range(1, 5):
            task_dataset = TaskDataset(dataset,
                                       task_transforms=[NWays(dataset, ways),
                                                        LoadData(dataset),
                                                        RemapLabels(dataset)],
                                       num_tasks=NUM_TASKS)
            for task in task_dataset:
                for label in range(ways):
                    self.assertTrue(label in task[1])


if __name__ == '__main__':
    unittest.main()
