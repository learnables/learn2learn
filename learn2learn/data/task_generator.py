import random
from collections import defaultdict
from itertools import permutations

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
        labels_to_indices, indices_to_labels = self.get_labels_indices()
        self.labels_to_indices = labels_to_indices
        self.indices_to_labels = indices_to_labels
        self.labels = list(self.labels_to_indices.keys())

    def __getitem__(self, item):
        return self.dataset[item]

    def __len__(self):
        return len(self.dataset)

    def get_labels_indices(self):
        """
        Iterates over the entire dataset and creates a map of target to indices.

        Returns: A dict with key as the label and value as list of indices.

        """

        assert hasattr(self.dataset, '__getitem__'), \
            'Requires iterable-style dataset.'

        labels_to_indices = defaultdict(list)
        indices_to_labels = defaultdict(list)
        for i in range(len(self.dataset)):
            try:
                # if label is a Tensor, then take get the scala value
                label = self.dataset[i][1].item()
            except AttributeError:
                # if label is a scalar then use as is
                label = self.dataset[i][1]
            except ValueError as e:
                raise ValueError(
                    'Requires scalar labels. \n' + str(e))

            labels_to_indices[label].append(i)
            indices_to_labels[i].append(label)
        return labels_to_indices, indices_to_labels


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
