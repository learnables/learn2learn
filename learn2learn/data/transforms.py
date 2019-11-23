#!/usr/bin/env python3

"""
Collection of general task transformations.
"""

import random
import collections


class LoadData(object):

    def __init__(self, dataset):
        self.dataset = dataset

    def __call__(self, task_description):
        for _, transforms in task_description:
            transforms.append(lambda x: self.dataset[x])
        return task_description


class NWays(object):

    def __init__(self, dataset, n=5):
        self.n = n
        self.dataset = dataset

    def __call__(self, task_description):
        classes = set([self.dataset.indices_to_labels[dt[0]] for dt in task_description])
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
