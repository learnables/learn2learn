import string
from unittest import TestCase

import numpy as np
import torch
from torch.utils.data import TensorDataset
from torchvision.datasets import MNIST
from numpy.testing import assert_array_equal

from learn2learn.data import MetaDataset
from learn2learn.vision.datasets import FullOmniglot


# Set up test data

class TempDataset():
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        return self.data[item], self.labels[item]


class TestDataset(TestCase):
    def setUp(self) -> None:
        download_location = "/tmp/datasets"

        # Create tensor and str label dataset
        self.n = 1500
        self.tensor_classes = [0, 1, 2, 3, 4]
        self.str_classes = ["0", "1", "2", "3", "4"]
        self.alphabets = list(string.ascii_lowercase)

        self.features = 10

        tensor_data = torch.from_numpy(np.random.randn(self.n, self.features))
        tensor_labels = torch.from_numpy(np.random.choice(self.tensor_classes, self.n))
        tensor_dataset = TensorDataset(tensor_data, tensor_labels)

        str_data = np.random.randn(self.n, self.features)
        str_labels = np.random.choice(self.str_classes, self.n)
        str_dataset = TempDataset(str_data, str_labels)

        alphabet_data = np.repeat(np.arange(26), self.features).reshape(-1, self.features)
        alphabet_dataset = TempDataset(alphabet_data, self.alphabets)

        # Use MNIST
        self.mnist_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        mnist = MNIST(download_location, train=True, download=True)

        # Use OMNIGLOT
        self.omniglot_classes = [i for i in range(1623)]
        omniglot = FullOmniglot(root=download_location, download=True)

        # Create Datasets
        self.meta_tensor_dataset = MetaDataset(tensor_dataset)
        self.meta_str_dataset = MetaDataset(str_dataset)
        self.meta_alpha_dataset = MetaDataset(alphabet_dataset)
        # self.mnist_dataset = MetaDataset(mnist)
        # self.omniglot_dataset = MetaDataset(omniglot)


class TestMetaDataset(TestDataset):

    def test_data_length(self):
        self.assertEqual(len(self.meta_tensor_dataset), self.n)
        self.assertEqual(len(self.meta_str_dataset), self.n)
        self.assertEqual(len(self.meta_alpha_dataset), 26)
        # self.assertEqual(len(self.mnist_dataset), 60000)
        # self.assertEqual(len(self.omniglot_dataset), 32460)

    def test_data_labels_length(self):
        self.assertEqual(len(self.meta_tensor_dataset.labels), len(self.tensor_classes))
        self.assertEqual(len(self.meta_str_dataset.labels), len(self.str_classes))
        self.assertEqual(len(self.meta_alpha_dataset.labels), 26)
        # self.assertEqual(len(self.mnist_dataset.labels), len(self.mnist_classes))
        # self.assertEqual(len(self.omniglot_dataset.labels), len(self.omniglot_classes))

    def test_data_labels_values(self):
        self.assertEqual(sorted(self.meta_tensor_dataset.labels), sorted(self.tensor_classes))
        self.assertEqual(sorted(self.meta_str_dataset.labels), sorted(self.str_classes))
        # self.assertEqual(sorted(self.meta_alpha_dataset.labels), sorted(self.alphabets))
        # self.assertEqual(sorted(self.omniglot_dataset.labels), sorted(self.omniglot_classes))

    def test_get_item(self):
        for i in range(5):
            rand_index = np.random.randint(0, 26)
            data, label = self.meta_alpha_dataset[rand_index]
            assert_array_equal(data, [rand_index for _ in range(self.features)])
            self.assertEqual(label, chr(97 + rand_index))

    def test_get_dict_of_labels_to_indices(self):
        dict_label_to_indices = self.meta_alpha_dataset.get_dict_of_labels_to_indices()
        self.assertEqual(sorted(list(dict_label_to_indices.keys())), self.alphabets)
        for key in dict_label_to_indices:
            self.assertEqual(dict_label_to_indices[key][0], ord(key)-97)
