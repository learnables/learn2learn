#!/usr/bin/env python3

import string

import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset
from torchvision.datasets import MNIST

from learn2learn.vision.datasets import FullOmniglot


class TempDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        return self.data[item], self.labels[item]


class TestDatasets():
    def __init__(self):
        self.download_location = "/tmp/datasets"
        self.n = 1500
        self.features = 10

        self.tensor_classes = [0, 1, 2, 3, 4]
        self.str_classes = ["0", "1", "2", "3", "4"]
        self.alphabets = list(string.ascii_lowercase)
        self.mnist_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.omniglot_classes = [i for i in range(1623)]

        tensor_data = torch.from_numpy(np.random.randn(self.n, self.features))
        tensor_labels = torch.from_numpy(np.random.choice(self.tensor_classes, self.n))

        str_data = np.random.randn(self.n, self.features)
        str_labels = np.random.choice(self.str_classes, self.n)

        alphabet_data = np.repeat(np.arange(26), self.features).reshape(-1, self.features)

        self.tensor_dataset = TensorDataset(tensor_data, tensor_labels)
        self.str_dataset = TempDataset(str_data, str_labels)
        self.alphabet_dataset = TempDataset(alphabet_data, self.alphabets)

    def get_mnist(self):
        return MNIST(self.download_location, train=True, download=True)

    def get_omniglot(self):
        return FullOmniglot(root=self.download_location, download=True)
