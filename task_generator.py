from collections import defaultdict

import numpy as np
from torch.utils.data import Dataset


class SampleDataset(Dataset):
    """
    SampleDataset to be used by TaskGenerator
    """

    def __init__(self, data, labels):
        self.data = data
        self.label = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]


class TaskGenerator:
    def __init__(self, dataset: Dataset, ways: int = 3, shots: int = 5):
        """

        Args:
            dataset: should be a Dataset that returns (data, target)
            ways: number of labels to sample from
            shots: sample size

        """

        self.dataset = dataset
        self.ways = ways
        self.shots = shots
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
        labels = list(self.target_to_indices.keys())
        return np.random.choice(labels, size=self.ways, replace=False)

    def sample(self, labels_to_sample=None):
        """ Returns a dataset and the labels that we have sampled.

        The dataset is of length `shots * ways`.
        The length of labels we have sampled is the same as `shots`.

        Args:
            labels_to_sample: List of labels you want to sample from

        Returns: Dataset, list(labels)

        """
        if labels_to_sample is None:
            labels_to_sample = self.get_random_label_pair()
        data_indices = []
        labels = []
        for label in labels_to_sample:
            data_indices.extend(np.random.choice(self.target_to_indices[label], self.shots, replace=False))
            labels.extend(np.full(self.shots, fill_value=label))

        data = [self.dataset[idx][0] for idx in data_indices]

        return SampleDataset(data, labels), labels_to_sample
