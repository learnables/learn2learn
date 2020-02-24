# cython: embedsignature=True
# cython: binding=True
# cython: language_version=3
#!/usr/bin/env python3

"""
**Description**

Collection of general task transformations.

A task transformation is an object that implements the callable interface.
(Either a function or an object that implements the `__call__` special method.)
Each transformation is called on a task description, which consists of a list of
`DataDescription` with attributes `index` and `transforms`, where
`index` corresponds to the index of single data sample inthe dataset, and `transforms` is a list of transformations
that will be applied to the sample.
Each transformation must return a new task description.

At first, the task description contains all samples from the dataset.
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

cimport cython
from cpython cimport array

import cython
import copy
import random
import collections
import functools
import array

from .task_dataset cimport DataDescription
from .task_dataset import DataDescription


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
        for data_description in task_description:
            data_description.transforms.append(lambda x: self.dataset[x])
        return task_description


cdef class FilterLabels(object):

    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/data/transforms.py)

    **Description**

    Removes samples that do not belong to the given set of labels.

    **Arguments**

    * **dataset** (Dataset) - The dataset from which to load the sample.
    * **labels** (list) - The list of labels to include.

    """

    cdef public:
        list labels
        int[:] filtered_indices

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __init__(self, dataset, list labels):
        cdef dict indices_to_labels = <dict>dataset.indices_to_labels
        cdef long len_dataset = len(dataset)
        cdef long i
        self.labels = labels
        self.filtered_indices = array.array('i', [0] * len_dataset)
        for i in range(len_dataset):
            self.filtered_indices[i] = int(indices_to_labels[i] in self.labels)
#        self.filtered_indices = array.array('i',
#                                            [bool(dataset.indices_to_labels[i] in self.labels)
#                                             for i in range(len(dataset))])

    def __call__(self, list task_description):
        cdef DataDescription dd
        cdef list result = []
        for dd in task_description:
            if self.filtered_indices[dd.index]:
                result.append(dd)
        return result


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
        pairs = [(dd, self.dataset.indices_to_labels[dd.index])
                 for dd in task_description]
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
        labels = list(set(self.dataset.indices_to_labels[dd.index] for dd in task_description))
        if self.shuffle:
            random.shuffle(labels)

        def mapping(x):
            return labels.index(x)

        for dd in task_description:
            remap = functools.partial(self.remap, mapping=mapping)
            dd.transforms.append(remap)
        return task_description


cdef class NWays(object):

    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/data/transforms.py)

    **Description**

    Keeps samples from N random labels present in the task description.

    **Arguments**

    * **dataset** (Dataset) - The dataset from which to load the sample.
    * **n** (int, *optional*, default=2) - Number of labels to sample from the task
        description's labels.

    """

    cdef public:
        int n
        dict indices_to_labels

    def __init__(self, dataset, n=2):
        self.n = n
        self.indices_to_labels = <dict>dataset.indices_to_labels

    def __call__(self, list task_description):
        cdef list classes = []
        cdef list result = []
        cdef set set_classes = set()
        cdef DataDescription dd
        for dd in task_description:
            set_classes.add(self.indices_to_labels[dd.index])
        classes = <list>set_classes
        classes = random.sample(classes, k=self.n)
        for dd in task_description:
            if self.indices_to_labels[dd.index] in classes:
                result.append(dd)
        return result
#        return [dd for dd in task_description if self.indices_to_labels[dd.index] in classes]


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
        for dd in task_description:
            cls = self.dataset.indices_to_labels[dd.index]
            class_to_data[cls].append(dd)
        if self.replacement:
            def sampler(x, k):
                return [copy.deepcopy(dd)
                        for dd in random.choices(x, k=k)]
        else:
            sampler = random.sample
        return sum([sampler(dds, k=self.k) for dds in class_to_data.values()], [])
