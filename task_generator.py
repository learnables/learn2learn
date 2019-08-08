from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset


class TaskGenerator():
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

    def get_dict_of_target_to_indices(self) -> dict(list):
        """ Iterates over the entire dataset and creates a map of target to indices.

        Returns: A dict with key as the label and value as list of indices.

        """
        target_to_indices = defaultdict(list)
        for i, (_, target) in enumerate(self.dataset):
            target_to_indices[target].append(i)
        return target_to_indices

    def get_random_label_pair(self):
        """ Creates a random label tuple.

        Selects `ways` number of random labels from the set of labels.

        Returns: list of labels

        """
        labels = self.target_to_indices.keys()
        return np.random.choice(labels, size=self.ways, replace=False)

    def sample(self):
        """ Returns a dataset. The dataset is of length `shots * ways`

        Returns: Dataset

        """
        labels_to_sample = self.get_random_label_pair()
        data = []
        labels = []
        for label in labels_to_sample:
            data = data.append(np.random.choice(self.target_to_indices[label], self.shots, replace=False))
            labels = np.full(self.shots, fill_value=label)
        return TensorDataset(torch.tensor(data), torch.tensor(labels))
