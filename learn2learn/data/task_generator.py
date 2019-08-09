from collections import defaultdict

import numpy as np
import torch as th
from torch.utils.data import Dataset


class SampleDataset(Dataset):
    """
    SampleDataset to be used by TaskGenerator
    """

    def __init__(self, data, labels, sampled_classes):
        self.data = data
        self.label = labels
        self.sampled_classes = sampled_classes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]


class LabelEncoder:
    def __init__(self, classes):
        """ Encodes a list of classes into indices, starting from 0.

        Args:
            classes: list, tuple of classes
        """
        # ensure we don't have duplicates in the list
        classes = sorted(list(set(classes)))
        new_class = 0
        self.class_to_idx = dict()
        self.idx_to_class = dict()
        for old_class in classes:
            self.class_to_idx.update({old_class: new_class})
            self.idx_to_class.update({new_class: old_class})
            new_class += 1


class TaskGenerator:
    def __init__(self, dataset: Dataset, ways: int = 3):
        """

        Args:
            dataset: should be a Dataset that returns (data, target)
            ways: number of labels to sample from

        """

        self.dataset = dataset
        self.ways = ways
        self.__len__: len(dataset)
        self.target_to_indices = self.get_dict_of_target_to_indices()

    def get_dict_of_target_to_indices(self):
        """ Iterates over the entire dataset and creates a map of target to indices.

        Returns: A dict with key as the label and value as list of indices.

        """
        target_to_indices = defaultdict(list)
        for i in range(len(self.dataset)):
            target_to_indices[self.dataset[i][1]].append(i)
        return target_to_indices

    def get_random_label_pair(self):
        """ Creates a random label tuple.

        Selects `ways` number of random labels from the set of labels.

        Returns: list of labels

        """
        classes = list(self.target_to_indices.keys())
        return np.random.choice(classes, size=self.ways, replace=False)

    def sample(self, shots: int = 5, classes_to_sample=None):
        """ Returns a dataset and the labels that we have sampled.

        The dataset is of length `shots * ways`.
        The length of labels we have sampled is the same as `shots`.

        Args:
            shots: sample size
            labels_to_sample: List of labels you want to sample from

        Returns: Dataset, list(labels)

        """
        if classes_to_sample is None:
            classes_to_sample = self.get_random_label_pair()
        label_encoding = LabelEncoder(classes_to_sample)
        data_indices = []
        classes = []
        for _class in classes_to_sample:
            data_indices.extend(np.random.choice(self.target_to_indices[_class], shots, replace=False))
            classes.extend(np.full(shots, fill_value=label_encoding.class_to_idx[_class]))

        data = [self.dataset[idx][0] for idx in data_indices]

        return SampleDataset(data, classes, classes_to_sample)
