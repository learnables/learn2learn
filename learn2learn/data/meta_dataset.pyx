# cython: language_version=3
#!/usr/bin/env python3

cimport cython

import os
import pickle
import random
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset
import learn2learn as l2l


class MetaDataset(Dataset):
    """

    **Description**

    Wraps a classification dataset to enable fast indexing of samples within classes.

    This class exposes two attributes specific to the wrapped dataset:

    * `labels_to_indices`: maps a class label to a list of sample indices with that label.
    * `indices_to_labels`: maps a sample index to its corresponding class label.

    Those dictionary attributes are often used to quickly create few-shot classification tasks.
    They can be passed as arguments upon instantiation, or automatically built on-the-fly.
    If the wrapped dataset has an attribute `_bookkeeping_path`, then the built attributes will be cached on disk and reloaded upon the next instantiation.
    This caching strategy is useful for large datasets (e.g. ImageNet-1k) where the first instantiation can take several hours.

    Note that if only one of `labels_to_indices` or `indices_to_labels` is provided, this class builds the other one from it.

    **Arguments**

    * **dataset** (Dataset) -  A torch Dataset.
    * **labels_to_indices** (dict, **optional**, default=None) -  A dictionary mapping labels to the indices of their samples.
    * **indices_to_labels** (dict, **optional**, default=None) -  A dictionary mapping sample indices to their corresponding label.

    **Example**
    ~~~python
    mnist = torchvision.datasets.MNIST(root="/tmp/mnist", train=True)
    mnist = l2l.data.MetaDataset(mnist)
    ~~~
    """

    def __init__(self, dataset, labels_to_indices=None, indices_to_labels=None):

        if not isinstance(dataset, Dataset):
            raise TypeError(
                "MetaDataset only accepts a torch dataset as input")

        self.dataset = dataset

        if hasattr(dataset, '_bookkeeping_path'):
            self.load_bookkeeping(dataset._bookkeeping_path)
        else:
            self.create_bookkeeping(
                labels_to_indices=labels_to_indices,
                indices_to_labels=indices_to_labels,
            )

    def __getitem__(self, item):
        return self.dataset[item]

    def __len__(self):
        return len(self.dataset)

    def create_bookkeeping(self, labels_to_indices=None, indices_to_labels=None):
        """
        Iterates over the entire dataset and creates a map of target to indices.

        Returns: A dict with key as the label and value as list of indices.
        """

        assert hasattr(self.dataset, '__getitem__'), \
            'Requires iterable-style dataset.'

        # Bootstrap from arguments
        if labels_to_indices is not None:
            indices_to_labels = {
                idx: label
                for label, indices in labels_to_indices.items()
                for idx in indices
            }
        elif indices_to_labels is not None:
            labels_to_indices = defaultdict(list)
            for idx, label in indices_to_labels.items():
                labels_to_indices[label].append(idx)
        else:  # Create from scratch
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


class UnionMetaDataset(MetaDataset):

    """

    **Description**

    Takes multiple MetaDataests and constructs their union.

    Note: The labels of all datasets are remapped to be in consecutive order.
    (i.e. the same label in two datasets will be to two different labels in the union)

    **Arguments**

    * **datasets** (list of Dataset) -  A list of torch Datasets.

    **Example**
    ~~~python
    train = torchvision.datasets.CIFARFS(root="/tmp/mnist", mode="train")
    train = l2l.data.MetaDataset(train)
    valid = torchvision.datasets.CIFARFS(root="/tmp/mnist", mode="validation")
    valid = l2l.data.MetaDataset(valid)
    test = torchvision.datasets.CIFARFS(root="/tmp/mnist", mode="test")
    test = l2l.data.MetaDataset(test)
    union = UnionMetaDataset([train, valid, test])
    assert len(union.labels) == 100
    ~~~
    """

    def __init__(self, datasets):
        datasets = [
            MetaDataset(ds) if not isinstance(ds, MetaDataset) else ds
            for ds in datasets
        ]
        self.datasets = datasets
        labels_to_indices = defaultdict(list)
        indices_to_labels = defaultdict(int)
        labels_count = 0
        indices_count = 0
        for dataset in datasets:
            labels_nooffset = {label: i for i, label in enumerate(dataset.labels)}
            for idx, label in dataset.indices_to_labels.items():
                label = labels_nooffset[label]
                indices_to_labels[indices_count + idx] = labels_count + label
                labels_to_indices[labels_count + label].append(indices_count + idx)
            indices_count += len(dataset.indices_to_labels)
            labels_count += len(dataset.labels_to_indices)

        self.labels_to_indices = labels_to_indices
        self.indices_to_labels = indices_to_labels
        self.labels = list(self.labels_to_indices.keys())

    def __getitem__(self, item):
        ds_count = 0
        for dataset in self.datasets:
            if ds_count + len(dataset) > item:
                data = list(dataset[item - ds_count])
                data[1] = self.indices_to_labels[item]
                return data
            ds_count += len(dataset)

    def __len__(self):
        return len(self.indices_to_labels)


class FilteredMetaDataset(MetaDataset):

    """

    **Description**

    Takes in a MetaDataset and filters it to only include a subset of its labels.

    Note: The labels of all datasets are **not** remapped.
    (i.e. the labels from the original dataset are retained)

    **Arguments**

    * **datasets** (Dataset) -  A torch Datasets.
    * **labels** (list of ints) - A list of labels to keep.

    **Example**
    ~~~python
    train = torchvision.datasets.CIFARFS(root="/tmp/mnist", mode="train")
    train = l2l.data.MetaDataset(train)
    filtered = FilteredMetaDataset(train, [4, 8, 2, 1, 9])
    assert len(filtered.labels) == 5
    ~~~
    """

    def __init__(self, dataset, labels):
        if not isinstance(dataset, MetaDataset):
            dataset = MetaDataset(dataset)
        self.dataset = dataset
        self.to_true_indices = []
        labels_to_indices = defaultdict(list)
        indices_to_labels = defaultdict(int)
        idx_count = 0
        for label in labels:
            for true_idx in dataset.labels_to_indices[label]:
                self.to_true_indices.append(true_idx)
                labels_to_indices[label].append(idx_count)
                indices_to_labels[idx_count] = dataset.indices_to_labels[true_idx]
                idx_count += 1

        self.labels_to_indices = labels_to_indices
        self.indices_to_labels = indices_to_labels
        self.labels = list(self.labels_to_indices.keys())

    def __getitem__(self, item):
        true_idx = self.to_true_indices[item]
        return self.dataset[true_idx]

    def __len__(self):
        return len(self.indices_to_labels)
