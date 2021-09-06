# cython: language_version=3
#!/usr/bin/env python3

"""
General wrapper to help create tasks.
"""

cimport cython
import random
import copy

from torch.utils.data import Dataset
from torch.utils.data._utils import collate

import learn2learn as l2l


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.nonecheck(False)
@cython.infer_types(False)
cdef list fast_allocate(long n):
    cdef list result = [None] * n
    cdef long i
    for i in range(n):
        result[i] = DataDescription(i)
    return result


cdef class DataDescription:

    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/data/task_dataset.py)

    **Description**

    Simple class to describe the data and its transforms in a task description.

    **Arguments**

    * **index** (int) - The index of the sample in the dataset.
    """

    def __init__(self, long index):
        self.index = index
        self.transforms = []


class Taskset(CythonTaskDataset):

    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/data/task_dataset.py)

    **Description**

    Creates a set of tasks from a given Dataset.

    In addition to the Dataset, TaskDataset accepts a list of task transformations (`task_transforms`)
    which define the kind of tasks sampled from the dataset.

    The tasks are lazily sampled upon indexing (or calling the `.sample()` method), and their
    descriptions cached for later use.
    If `num_tasks` is -1, the TaskDataset will not cache task descriptions and instead continuously resample
    new ones.
    In this case, the length of the TaskDataset is set to 1.

    For more information on tasks and task descriptions, please refer to the
    documentation of task transforms.

    **Arguments**

    * **dataset** (Dataset) - Dataset of data to compute tasks.
    * **task_transforms** (list, *optional*, default=None) - List of task transformations.
    * **num_tasks** (int, *optional*, default=-1) - Number of tasks to generate.

    **Example**
    ~~~python
    dataset = l2l.data.MetaDataset(MyDataset())
    transforms = [
        l2l.data.transforms.NWays(dataset, n=5),
        l2l.data.transforms.KShots(dataset, k=1),
        l2l.data.transforms.LoadData(dataset),
    ]
    taskset = TaskDataset(dataset, transforms, num_tasks=20000)
    for task in taskset:
        X, y = task
    ~~~
    """

    def __init__(self, dataset, task_transforms=None, num_tasks=-1, task_collate=None):
        super(Taskset, self).__init__(
            dataset=dataset,
            task_transforms=task_transforms,
            num_tasks=num_tasks,
            task_collate=task_collate,
        )


class TaskDataset(Taskset):

    def __init__(self, *args, **kwargs):
        super(TaskDataset, self).__init__(*args, **kwargs)
        l2l.utils.warn_once(
            message='TaskDataset is deprecated, use Taskset instead.',
            severity='deprecation',
        )


cdef class CythonTaskDataset:

    cdef public:
        object dataset
        object task_transforms
        object task_collate
        dict sampled_descriptions
        int num_tasks
        int _task_id

    def __init__(self, dataset, task_transforms=None, int num_tasks=-1, task_collate=None):
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
        self.task_collate = task_collate
        self._task_id = 0

    cpdef sample_task_description(self):
        #  Samples a new task description.
        # cdef list description = fast_allocate(len(self.dataset))
        description = None
        if callable(self.task_transforms):
            return self.task_transforms(description)
        for transform in self.task_transforms:
            description = transform(description)
        return description

    def get_task(self, task_description):
        #  Given a task description, creates the corresponding batch of data.
        all_data = []
        for data_description in task_description:
            data = data_description.index
            for transform in data_description.transforms:
                data = transform(data)
            all_data.append(data)
        return self.task_collate(all_data)

    def sample(self):
        """
        **Description**

        Randomly samples a task from the TaskDataset.

        **Example**
        ~~~python
        X, y = taskset.sample()
        ~~~
        """
        i = random.randint(0, len(self) - 1)
        return self[i]

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
