#!/usr/bin/env python3

"""
General wrapper to help create tasks.
"""

import sys
import copy

from torch.utils.data import Dataset
from torch.utils.data._utils import collate

import learn2learn as l2l


class TaskDataset(Dataset):

    def __init__(self, dataset, task_transforms=None, num_tasks=-1, task_collate=None):
        if not isinstance(dataset, l2l.data.MetaDataset):
            dataset = l2l.data.MetaDataset(dataset)
        if task_transforms is None:
            task_transforms = []
        if task_collate is None:
            task_collate = collate.default_collate
        if num_tasks < -1 or num_tasks == 0:
            raise ValueError('num_tasks needs to be -1 (infinity) or positive.')
        self.dataset = dataset
        self.num_tasks = num_tasks
        self.task_transforms = task_transforms
        self.sampled_descriptions = {}  # Maps indices to tasks' description dict
        self.dataset_description = [(i, []) for i in range(len(dataset))]
        self.task_collate = task_collate
        self._task_id = 0

    def sample_task_description(self):
        # Samples a new task description
        description = copy.deepcopy(self.dataset_description)
        for transform in self.task_transforms:
            description = transform(description)
        return description

    def get_task(self, task_description):
        # Given a task description, creates the corresponding batch of data
        all_data = []
        for data, transforms in task_description:
            for transform in transforms:
                data = transform(data)
            all_data.append(data)
        return self.task_collate(all_data)

    def __len__(self):
        if self.num_tasks == -1:
            # Ok to return 1, since __iter__ will run forever
            # and __getitem__ will always resample.
            return 1
        return self.num_tasks

    def __getitem__(self, i):
        if self.num_tasks == -1:
            return self.get_task(self.sample_task_description())
        if i not in self.sampled_descriptions:
            self.sampled_descriptions[i] = self.sample_task_description()
        task_description = self.sampled_descriptions[i]
        return self.get_task(task_description)

    def __iter__(self):
        self._task_id = 0
        return self

    def __next__(self):
        if self.num_tasks == -1:
            return self.get_task(self.sample_task_description())

        if self._task_id < self.num_tasks:
            task = self[self._task_id]
            self._task_id += 1
            return task
        else:
            raise StopIteration

    def __add__(self, other):
        msg = 'Adding datasets not yet supported for TaskDatasets.'
        raise NotImplementedError(msg)
