import random
from collections import defaultdict

import numpy as np
from torch.utils.data import Dataset


class SampleDataset(Dataset):
    """
    SampleDataset to be used by TaskGenerator
    """

    def __init__(self, data, labels, sampled_task):
        self.data = data
        self.label = labels
        self.sampled_task = sampled_task

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]


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
        self.dataset = dataset
        self.labels_to_indices = self.get_dict_of_labels_to_indices()
        self.labels = list(self.labels_to_indices.keys())

    def __getitem__(self, item):
        return self.dataset[item]

    def __len__(self):
        return len(self.dataset)

    def get_dict_of_labels_to_indices(self):
        """ Iterates over the entire dataset and creates a map of target to indices.

        Returns: A dict with key as the label and value as list of indices.

        """

        classes_to_indices = defaultdict(list)
        for i in range(len(self.dataset)):
            classes_to_indices[self.dataset[i][1]].append(i)
        return classes_to_indices


class LabelEncoder:
    def __init__(self, classes):
        """ Encodes a list of classes into indices, starting from 0.

        Args:
            classes: List of classes
        """
        assert len(set(classes)) == len(classes), "Classes contains duplicate values"
        self.class_to_idx = dict()
        self.idx_to_class = dict()
        for idx, old_class in enumerate(classes):
            self.class_to_idx.update({old_class: idx})
            self.idx_to_class.update({idx: old_class})


class TaskGenerator:
    """

    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/data/task_generator.py)

    **Description**

    A wrapper to generate few-shot classification tasks.



    `tasks` can both indicate predefined tasks, or just the number of tasks to sample.
    If specified as an int, a list of size `task` would be generated from which we'll sample.
    If specified as a list, then that list of tasks would be used to sample always.

    The acceptable shape of list would be `n * w`, with n the number of tasks to sample and w the number of ways.

    Each of the task should have w distinct elements all of which are required to be a subset of ways.

    **Arguments**

    * **dataset** (MetaDataset or Dataset) - The (meta-) dataset to wrap.
    * **classes** (list, *optional*, default=None) - List of classes to sample from,
        if none then sample from all available classes in dataset. (default: None)
    * **ways** (int, *optional*, default=2) - Number of labels to sample from.
    * **shots** (int, *optional*, default=1) - Number of data points per task to sample.
    * **tasks** (int or list, *optional*, default=1) - Tasks to be generated.
    """

    def __init__(self, dataset, classes=None, ways=2, tasks=1, shots=1):
        self.dataset = dataset
        self.ways = ways
        self.classes = classes
        self.shots = shots

        if not isinstance(dataset, MetaDataset):
            dataset = MetaDataset(dataset)

        if classes is None:
            self.classes = self.dataset.labels

        if isinstance(tasks, int):
            self.tasks = self.generate_n_tasks(tasks)
        elif isinstance(tasks, list):
            self.tasks = tasks
        else:
            raise TypeError(f"tasks is not either of int/list but rather {type(tasks)}")

        # used for next(taskgenerator)
        self.tasks_idx = 0

        assert len(self.classes) >= ways, ValueError("Ways are more than the number of classes available")
        self._check_classes(self.classes)
        self._check_tasks(self.tasks)

        # TODO : assert that shots are always less than equal to min_samples for each class

    def generate_n_tasks(self, n):
        # Args:
        #     n: Number of tasks to generate

        # Returns: A list of shape `n * w` where n is the number of tasks to generate and w is the ways.

        def get_samples():
            random.shuffle(self.classes)
            return self.classes[:self.ways]

        return [get_samples() for _ in range(n)]

    def __iter__(self):
        self.tasks_idx = 0
        return self

    def __len__(self):
        return len(self.tasks)

    def __next__(self):
        # TODO : Add the following test case
        # for i, task in enumerate(tg):
        #     assert task.sampled_task == tg.tasks[i]
        # Returns:
        try:
            task = self.sample(self.tasks[self.tasks_idx])
        except IndexError:
            raise StopIteration()

        self.tasks_idx += 1
        return task

    def sample(self, shots=None, task=None):
        """

        **Description**

        Returns a dataset and the labels that we have sampled.

        The dataset is of length `shots * ways`.
        The length of labels we have sampled is the same as `shots`.

        **Arguments**

        **shots** (int, *optional*, default=None) - Number of data points to return per class, if None gets self.shots.
        **task** (list, *optional*, default=None) - List of labels you want to sample from.

        **Returns**
        
        * Dataset - Containing the sampled task.

        """
        # If shots isn't defined, then try to inherit from object
        if shots is None:
            if self.shots is None:
                raise ValueError(
                    "Shots is undefined in object definition neither while calling the sample method.")
            shots = self.shots

        # If classes aren't specified while calling the function, then we can
        # sample from all the classes mentioned during the initialization of the TaskGenerator
        if task is None:
            # select few classes that will be selected for this task (for eg, 6,4,7 from 0-9 in MNIST when ways are 3)
            rand_idx = random.randint(0, len(self.tasks) - 1)
            task_to_sample = self.tasks[rand_idx]
        else:
            task_to_sample = task
            assert self._check_task(task_to_sample), ValueError("Task is malformed.")

        # encode labels (map 6,4,7 to 0,1,2 so that we can do a BCELoss)
        label_encoder = LabelEncoder(task_to_sample)

        data_indices = []
        data_labels = []
        for _class in task_to_sample:
            # select subset of indices from each of the classes and add it to data_indices
            data_indices.extend(np.random.choice(self.dataset.labels_to_indices[_class], shots, replace=False))
            # add those labels to data_labels (6 mapped to 0, so add 0's initially then 1's (for 4) and so on)
            data_labels.extend(np.full(shots, fill_value=label_encoder.class_to_idx[_class]))

        # map data indices to actual data
        data = [self.dataset[idx][0] for idx in data_indices]
        return SampleDataset(data, data_labels, task_to_sample)

    def _check_classes(self, classes):
        """ ensure that classes are a subset of dataset.labels """
        assert len(set(classes) - set(self.dataset.labels)) == 0, "classes contains a label that isn't in dataset"

    def _check_task(self, task) -> bool:
        """ check if each individual task is a subset of self.classes and has no duplicates """
        return (len(set(task) - set(self.classes)) == 0) and (len(set(task)) - len(task) == 0)

    def _check_tasks(self, tasks):
        """ ensure that all tasks are correctly defined. """
        invalid_tasks = list(filter(lambda task: not self._check_task(task), tasks))
        assert len(invalid_tasks) == 0, f"Following task in mentioned tasks are unacceptable. \n {invalid_tasks}"
