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
        # shuffle all our classes
        np.random.shuffle(classes)
        assert len(set(classes)) == len(classes), "Classes contains duplicate values"
        self.class_to_idx = dict()
        self.idx_to_class = dict()
        for idx, old_class in enumerate(classes):
            self.class_to_idx.update({old_class: idx})
            self.idx_to_class.update({idx: old_class})


class TaskGenerator:
    def __init__(self, dataset: MetaDataset, classes: list = None, ways: int = 3):
        """

        Args:
            dataset: should be a MetaDataset
            classes: List of classes to sample from
            ways: number of labels to sample from
        """

        # TODO : Add conditional check to work even when dataset isn't MetaDataset and a torch.Dataset
        #  then also it should work
        self.dataset = dataset
        self.ways = ways
        self.classes = classes
        if classes is None:
            self.classes = self.dataset.labels

        assert len(self.classes) >= ways, ValueError("Ways are more than the number of classes available")
        self._check_classes(self.classes)

    def sample(self, classes: list = None, shots: int = 5):
        """ Returns a dataset and the labels that we have sampled.

        The dataset is of length `shots * ways`.
        The length of labels we have sampled is the same as `shots`.

        Args:
            shots: sample size
            classes: Optional list,
            labels_to_sample: List of labels you want to sample from

        Returns: Dataset, list(labels)

        """
        classes = self.classes if classes is None else classes
        self._check_classes(classes)
        classes_to_sample = np.random.choice(classes, size=self.ways, replace=False)
        label_encoding = LabelEncoder(classes_to_sample)
        data_indices = []
        classes = []
        for _class in classes_to_sample:
            data_indices.extend(np.random.choice(self.dataset.labels_to_indices[_class], shots, replace=False))
            classes.extend(np.full(shots, fill_value=label_encoding.class_to_idx[_class]))

        data = [self.dataset[idx][0] for idx in data_indices]

        return SampleDataset(data, classes, classes_to_sample)

    def _check_classes(self, classes):
        assert len(set(classes) - set(self.dataset.labels)) == 0, "classes contains a label that isn't in dataset"
