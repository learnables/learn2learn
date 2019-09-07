import random
from collections import defaultdict
from typing import Optional, List, Union

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
    """ It wraps a torch dataset by creating a map of target to indices.
    This comes in handy when we want to sample elements randomly for a particular label.

    Args:
        dataset: A torch dataset

    Notes:
        For l2l to work its important that the dataset returns a (data, target) tuple.
        If your dataset doesn't return that, it should be trivial to wrap your dataset
        with another class to do that.
        #TODO : Add example for wrapping a non standard l2l dataset
    """

    def __init__(self, dataset: Dataset):
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
    def __init__(self, dataset: MetaDataset,
                 classes: Optional[list] = None,
                 ways: int = 3,
                 shots: Optional[int] = 1,
                 tasks: Union[List[int], int] = 10):
        """

        Args:
            dataset: should be a MetaDataset.
            classes: List of classes to sample from, if none then sample from all available classes in dataset.
                    (default: None)
            ways: number of labels to sample from (default: 3)
            shots: number of data points per task to sample (default: 1)
            task: if specified as an int, a list of of size task would be generated from which we'll sample.
                    if specified as a list, then that list of tasks would be used to sample always.

                    The acceptable shape of list would be `n * w`
                    n : the number of tasks to sample
                    w : the number of ways

                    Each of the task should have w distinct elements all of which are required to be a subset of ways.
        """

        # TODO : Add conditional check to work even when dataset isn't MetaDataset and a torch.Dataset
        #  then also it should work
        self.dataset = dataset
        self.ways = ways
        self.classes = classes
        self.shots = shots

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
        """

        Args:
            n: Number of tasks to generate

        Returns: A list of shape `n * w` where n is the number of tasks to generate and w is the ways.

        """
        random.shuffle(self.classes)
        return [self.classes[:self.ways] for _ in range(n)]

    def __next__(self):
        try:
            task = self.sample(self.tasks[self.tasks_idx])
        except IndexError:
            raise StopIteration()

        self.tasks_idx += 1
        return task

    def sample(self, task: list = None, shots: Optional[int] = None):
        """ Returns a dataset and the labels that we have sampled.

        The dataset is of length `shots * ways`.
        The length of labels we have sampled is the same as `shots`.

        Args:
            shots: number of data points to return per class, if none then gets the (default : None)
            classes: Optional list,
            labels_to_sample: List of labels you want to sample from

        Returns: Dataset, list(labels)

        Raises:
            ValueError : when shots is undefined both in class definition and method

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
            task_to_sample = np.random.choice(self.tasks, size=1)
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
        assert len(set(classes) - set(self.dataset.labels)) == 0, "classes contains a label that isn't in dataset"

    def _check_task(self, task):
        # check if each individual task is a subset of self.classes and has no duplicates
        return (set(task) - set(self.classes) == 0) and (len(set(task)) - len(task) == 0)

    def _check_tasks(self, tasks):
        # ensure that all tasks are correctly defined.
        assert len(list(filter(self._check_task, tasks))) - len(
            tasks) == 0, "Some task in mentioned tasks contain extra variables"
