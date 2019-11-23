#!/usr/bin/env python3

import os
import pickle
import random
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset


class MetaDataset(Dataset):
    """

    **Descritpion**

    It wraps a torch dataset by creating a map of target to indices.
    This comes in handy when we want to sample elements randomly for a particular label.

    Notes:
        For l2l to work its important that the dataset returns a (data, target) tuple.
        If your dataset doesn't return that, it should be trivial to wrap your dataset
        with another class to do that.
        #TODO : Add example for wrapping a non standard l2l dataset

    **Arguments**

    * **dataset** (Dataset) -  A torch dataset.
    * **labels_to_indices** (Dict) -  A dictionary mapping label to their indices.
                                     If not specified then we loop through all the datapoints to understand the mapping. (default: None)

    **Example**
    ~~~python
    mnist = torchvision.datasets.MNIST(root="/tmp/mnist", train=True)
    mnist = l2l.data.MetaDataset(mnist)
    ~~~
    """

    def __init__(self, dataset):

        if not isinstance(dataset, Dataset):
            raise TypeError(
                "MetaDataset only accepts a torch dataset as input")

        self.dataset = dataset

        if hasattr(dataset, '_bookkeeping_path'):
            self.load_bookkeeping(dataset._bookkeeping_path)
        else:
            self.create_bookkeeping()

    def __getitem__(self, item):
        return self.dataset[item]

    def __len__(self):
        return len(self.dataset)

    def create_bookkeeping(self):
        """
        Iterates over the entire dataset and creates a map of target to indices.

        Returns: A dict with key as the label and value as list of indices.
        """

        assert hasattr(self.dataset, '__getitem__'), \
            'Requires iterable-style dataset.'

        labels_to_indices = defaultdict(list)
        indices_to_labels = defaultdict(int)
        for i in range(len(self.dataset)):
            try:
                label = self.dataset[i][1]
                # if label is a Tensor, then take get the scalar value
                if hasattr(label, 'item'):
                    label = self.dataset[i][1].item()
            except ValueError as e:
                raise ValueError(
                    'Requires scalar labels. \n' + str(e))

            labels_to_indices[label].append(i)
            indices_to_labels[i] = label

        self.labels_to_indices = labels_to_indices
        self.indices_to_labels = indices_to_labels
        self.labels = list(self.labels_to_indices.keys())

        self._bookkeeping = {
            'labels_to_indices': self.labels_to_indices,
            'indices_to_labels': self.indices_to_labels,
            'labels': self.labels
        }

    def load_bookkeeping(self, path):
        if not os.path.exists(path):
            self.create_bookkeeping()
            self.serialize_bookkeeping(path)
        else:
            with open(path, 'rb') as f:
                self._bookkeeping = pickle.load(f)
            self.labels_to_indices = self._bookkeeping['labels_to_indices']
            self.indices_to_labels = self._bookkeeping['indices_to_labels']
            self.labels = self._bookkeeping['labels']

    def serialize_bookkeeping(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self._bookkeeping, f, protocol=-1)


class NShotKWayTaskSampler():

    def __init__(self, label, episodes, ways, shots, query, fixed_classes=None):
        self.episodes = episodes
        self.ways = ways
        self.fixed_classes = fixed_classes
        self.total_subset_len = shots + query
        label = torch.Tensor(label).int()

        if fixed_classes is not None:
            raise ValueError(
                "Currently fixed classes not supported. Will be supported in a week! ;)")

        # TODO: Need to add support for fixed classes
        if shots < 1:
            raise ValueError('shots have to be greater than 1.')
        if ways > len(torch.unique(label)):
            raise ValueError(
                'ways has to be less than number of unique labels')

        self.index_list = []
        for i in range(max(label) + 1):
            indices = (label == i).nonzero().reshape(-1)
            self.index_list.append(indices)

    def __len__(self):
        return self.episodes

    def __iter__(self):
        for i_batch in range(self.episodes):
            batch = []
            if self.fixed_classes is None:
                classes = torch.randperm(len(self.index_list))[:self.ways]
                for class_id in classes:
                    class_subset = self.index_list[class_id]
                    pos = torch.randperm(len(class_subset))[
                        :self.total_subset_len]
                    batch.append(class_subset[pos])
                batch = torch.stack(batch).t().reshape(-1)
            yield batch
