#!/usr/bin/env python

import random
import torch
import learn2learn as l2l


class TasksetSampler(torch.utils.data.Sampler):

    def __init__(self, taskset, shuffle=True):
        self.taskset = taskset
        self.shuffle = shuffle

    def description2indices(self, task_description):
        return [dd.index for dd in task_description]

    def __iter__(self):
        if self.taskset.num_tasks == -1:  # loop infinitely
            while True:
                yield self.description2indices(
                    self.taskset.sample_task_description()
                )
        else:  # loop over the range of tasks
            task_indices = list(range(self.taskset.num_tasks))
            if self.shuffle:
                random.shuffle(task_indices)
            for i in task_indices:
                if i not in self.taskset.sampled_descriptions:
                    self.taskset.sampled_descriptions[i] = self.taskset.sample_task_description()
                yield self.description2indices(
                    self.taskset.sampled_descriptions[i]
                )


if __name__ == "__main__":
    NUM_TASKS = 10
    NUM_DATA = 128
    X_SHAPE = 16
    Y_SHAPE = 10
    EPSILON = 1e-6
    SUBSET_SIZE = 5
    WORKERS = 4
    META_BSZ = 16
    data = torch.randn(NUM_DATA, X_SHAPE)
    labels = torch.randint(0, Y_SHAPE, (NUM_DATA, ))
    dataset = torch.utils.data.TensorDataset(data, labels)
    dataset = l2l.data.MetaDataset(dataset)
    taskset = l2l.data.Taskset(
        dataset,
        task_transforms=[
            l2l.data.transforms.FusedNWaysKShots(dataset, n=2, k=1),
            l2l.data.transforms.LoadData(dataset),
            l2l.data.transforms.RemapLabels(dataset),
            l2l.data.transforms.ConsecutiveLabels(dataset),
        ],
        num_tasks=NUM_TASKS,
    )

    sampler = TasksetSampler(taskset)
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_sampler=sampler,
    )
    for task in dataloader:
        print(task)

    __import__('pdb').set_trace()
