from collections import defaultdict

import numpy as np
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
    def __init__(self, dataset: Dataset, ways: int = 3, split=False, test_size=3, test_classes=[]):
        """

        Args:
            dataset: should be a Dataset that returns (data, target)
            ways: number of labels to sample from
            split: If set to True, it will split the labels into train and test.
            test_size: If split is set to True, then it uses test_size samples in test

        """

        self.dataset = dataset
        self.ways = ways
        self.split = split

        self.__len__: len(dataset)
        self.target_to_indices = self.get_dict_of_target_to_indices()

        self.classes = list(self.target_to_indices.keys())
        self.train_classes = list()
        self.test_classes = test_classes

        if self.split:
            self.split_datasets(n=test_size)

    def split_datasets(self, n):
        """ This method would randomly select n classes to belong in test classes.
        This way we can create two TaskGenerators and measure how easily
        we learn on data we've never seen before.

        """
        if self.split:
            if len(self.test_classes) > 0:
                self.test_classes = self.test_classes
            else:
                self.test_classes = np.random.choice(self.classes, size=n, replace=False)

            self.train_classes = list(set(self.classes) - set(self.test_classes))

    def get_dict_of_target_to_indices(self):
        """ Iterates over the entire dataset and creates a map of target to indices.

        Returns: A dict with key as the label and value as list of indices.

        """
        target_to_indices = defaultdict(list)
        for i in range(len(self.dataset)):
            target_to_indices[self.dataset[i][1]].append(i)
        return target_to_indices

    def get_random_label_pair(self, sample_from_train):
        """ Creates a random label tuple.

        Selects `ways` number of random labels from the set of labels.

        Returns: list of labels

        """
        if self.split:
            if sample_from_train:
                return np.random.choice(self.train_classes, size=self.ways, replace=False)
            else:
                return np.random.choice(self.test_classes, size=self.ways, replace=False)
        else:
            return np.random.choice(self.classes, size=self.ways, replace=False)

    def sample(self, shots: int = 5, classes_to_sample=None, sample_from_train=True):
        """ Returns a dataset and the labels that we have sampled.

        The dataset is of length `shots * ways`.
        The length of labels we have sampled is the same as `shots`.

        Args:
            shots: sample size
            labels_to_sample: List of labels you want to sample from

        Returns: Dataset, list(labels)

        """
        if classes_to_sample is None:
            classes_to_sample = self.get_random_label_pair(sample_from_train)
        label_encoding = LabelEncoder(classes_to_sample)
        data_indices = []
        classes = []
        for _class in classes_to_sample:
            data_indices.extend(np.random.choice(self.target_to_indices[_class], shots, replace=False))
            classes.extend(np.full(shots, fill_value=label_encoding.class_to_idx[_class]))

        data = [self.dataset[idx][0] for idx in data_indices]

        return SampleDataset(data, classes, classes_to_sample)
