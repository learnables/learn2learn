#!/usr/bin/env python3

"""
Collection of general task transformations.
"""

import random
import collections
import functools


class LoadData(object):

    def __init__(self, dataset):
        self.dataset = dataset

    def __call__(self, task_description):
        for _, transforms in task_description:
            transforms.append(lambda x: self.dataset[x])
        return task_description


class FilterLabels(object):

    def __init__(self, dataset, labels):
        self.dataset = dataset
        self.labels = labels

    def __call__(self, task_description):
        return [d for d in task_description
                if self.dataset.indices_to_labels[d[0]] in self.labels]

class ConsecutiveLabels(object):

    def __init__(self, dataset):
        self.dataset = dataset

    def __call__(self, task_description):
        pairs = [(data, self.dataset.indices_to_labels[data[0]])
                 for data in task_description]
        pairs = sorted(pairs, key=lambda x: x[1])
        return [p[0] for p in pairs]


class RemapLabels(object):

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
        mapping = lambda x: labels.index(x)
        for data in task_description:
            remap = functools.partial(self.remap, mapping=mapping)
            data[1].append(remap)
        return task_description


class NWays(object):

    def __init__(self, dataset, n=5):
        self.n = n
        self.dataset = dataset

    def __call__(self, task_description):
        classes = list(set([self.dataset.indices_to_labels[dt[0]] for dt in task_description]))
        classes = random.sample(classes, k=self.n)
        return [data for data in task_description if self.dataset.indices_to_labels[data[0]] in classes]


class KShots(object):

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
