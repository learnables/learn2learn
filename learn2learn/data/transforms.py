#!/usr/bin/env python3

"""
**Description**

Collection of general task transformations.

A task transformation is an object that implements the callable interface.
(Either a function or an object that implements the `__call__` special method.)
Each transformation is called on a task description, which consists of a list of
tuples of the form: `[(sample1, transforms1), (sample2, transforms2) ...]`, where
`sample` corresponds to a single data sample, and `transforms` is a list of transformations
that will be applied to the sample.
Each transformation must return a new task description.

At first, the task description contains all samples from the dataset.
The `sample`s are represented as their integer index in the dataset, and their corresponding
`transform`s is empty.
A task transform takes this task description list and modifies it such that a particular task is
created.
For example, the `NWays` task transform filters data samples from the task description
such that remaining ones belong to a random subset of all classes available.
(The size of the subset is controlled via the class's `n` argument.)
On the other hand, the `LoadData` task transform simply appends a call to load the actual
data from the dataset to the list of transformations of each sample.

To create a task from a task description, the `TaskDataset` applies each sample's list of `transform`s
in order.
Then, all samples are collated via the `TaskDataset`'s collate function.

"""

import random
import collections
import functools


class LoadData(object):

    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/data/transforms.py)

    **Description**

    Loads a sample from the dataset given its index.

    **Arguments**

    * **dataset** (Dataset) - The dataset from which to load the sample.

    """

    def __init__(self, dataset):
        self.dataset = dataset

    def __call__(self, task_description):
        for _, transforms in task_description:
            transforms.append(lambda x: self.dataset[x])
        return task_description


class FilterLabels(object):

    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/data/transforms.py)

    **Description**

    Removes samples that do not belong to the given set of labels.

    **Arguments**

    * **dataset** (Dataset) - The dataset from which to load the sample.
    * **labels** (list) - The list of labels to include.

    """

    def __init__(self, dataset, labels):
        self.dataset = dataset
        self.labels = labels

    def __call__(self, task_description):
        return [d for d in task_description
                if self.dataset.indices_to_labels[d[0]] in self.labels]


class ConsecutiveLabels(object):

    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/data/transforms.py)

    **Description**

    Re-orders the samples in the task description such that they are sorted in
    consecutive order.

    Note: when used before `RemapLabels`, the labels will be homogeneously clustered,
    but in no specific order.

    **Arguments**

    * **dataset** (Dataset) - The dataset from which to load the sample.

    """

    def __init__(self, dataset):
        self.dataset = dataset

    def __call__(self, task_description):
        pairs = [(data, self.dataset.indices_to_labels[data[0]])
                 for data in task_description]
        pairs = sorted(pairs, key=lambda x: x[1])
        return [p[0] for p in pairs]


class RemapLabels(object):

    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/data/transforms.py)

    **Description**

    Given samples from K classes, maps the labels to 0, ..., K.

    **Arguments**

    * **dataset** (Dataset) - The dataset from which to load the sample.

    """

    def __init__(self, dataset, shuffle=True):
        self.dataset = dataset
        self.shuffle = shuffle

    def remap(self, data, mapping):
        data = [d for d in data]
        data[1] = mapping(data[1])
        return data

    def __call__(self, task_description):
        labels = list(set(self.dataset.indices_to_labels[d[0]] for d in task_description))
        if self.shuffle:
            random.shuffle(labels)

        def mapping(x):
            return labels.index(x)

        for data in task_description:
            remap = functools.partial(self.remap, mapping=mapping)
            data[1].append(remap)
        return task_description


class NWays(object):

    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/data/transforms.py)

    **Description**

    Keeps samples from N random labels present in the task description.

    **Arguments**

    * **dataset** (Dataset) - The dataset from which to load the sample.
    * **n** (int, *optional*, default=2) - Number of labels to sample from the task
        description's labels.

    """

    def __init__(self, dataset, n=2):
        self.n = n
        self.dataset = dataset

    def __call__(self, task_description):
        classes = list(set([self.dataset.indices_to_labels[dt[0]] for dt in task_description]))
        classes = random.sample(classes, k=self.n)
        return [data for data in task_description if self.dataset.indices_to_labels[data[0]] in classes]


class KShots(object):

    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/data/transforms.py)

    **Description**

    Keeps K samples for each present labels.

    **Arguments**

    * **dataset** (Dataset) - The dataset from which to load the sample.
    * **k** (int, *optional*, default=1) - The number of samples per label.
    * **replacement** (bool, *optional*, default) - Whether to sample with replacement.

    """

    def __init__(self, dataset, k=1, replacement=False):
        self.dataset = dataset
        self.k = k
        self.replacement = replacement

    def __call__(self, task_description):
        # TODO: The order of the data samples is not preserved.
        class_to_data = collections.defaultdict(list)
        for data in task_description:
            cls = self.dataset.indices_to_labels[data[0]]
            class_to_data[cls].append(data)
        if self.replacement:
            sampler = random.choices
        else:
            sampler = random.sample
        return sum([sampler(datas, k=self.k) for datas in class_to_data.values()], [])
