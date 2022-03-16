#!/usr/bin/env python3

import unittest
from unittest import TestCase

import learn2learn as l2l
import numpy as np
from numpy.testing import assert_array_equal

from learn2learn.data import MetaDataset
from .util_datasets import TestDatasets


class TestMetaDataset(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ds = TestDatasets()
        cls.meta_tensor_dataset = MetaDataset(cls.ds.tensor_dataset)
        cls.meta_str_dataset = MetaDataset(cls.ds.str_dataset)
        cls.meta_alpha_dataset = MetaDataset(cls.ds.alphabet_dataset)
        cls.omniglot_dataset = MetaDataset(cls.ds.get_omniglot())

    def test_fails_with_non_torch_dataset(self):
        try:
            MetaDataset(np.random.randn(100, 100))
            return False
        except TypeError:
            return True
        finally:
            return False

    def test_data_length(self):
        self.assertEqual(len(self.meta_tensor_dataset), self.ds.n)
        self.assertEqual(len(self.meta_str_dataset), self.ds.n)
        self.assertEqual(len(self.meta_alpha_dataset), 26)
        self.assertEqual(len(self.omniglot_dataset), 32460)

    def test_data_labels_length(self):
        self.assertEqual(len(self.meta_tensor_dataset.labels), len(self.ds.tensor_classes))
        self.assertEqual(len(self.meta_str_dataset.labels), len(self.ds.str_classes))
        self.assertEqual(len(self.meta_alpha_dataset.labels), 26)
        self.assertEqual(len(self.omniglot_dataset.labels), len(self.ds.omniglot_classes))

    def test_data_labels_values(self):
        self.assertEqual(sorted(self.meta_tensor_dataset.labels), sorted(self.ds.tensor_classes))
        self.assertEqual(sorted(self.meta_str_dataset.labels), sorted(self.ds.str_classes))
        self.assertEqual(sorted(self.meta_alpha_dataset.labels), sorted(self.ds.alphabets))
        self.assertEqual(sorted(self.omniglot_dataset.labels), sorted(self.ds.omniglot_classes))

    def test_get_item(self):
        for i in range(5):
            rand_index = np.random.randint(0, 26)
            data, label = self.meta_alpha_dataset[rand_index]
            assert_array_equal(data, [rand_index for _ in range(self.ds.features)])
            self.assertEqual(label, chr(97 + rand_index))

    def test_labels_to_indices(self):
        self.meta_alpha_dataset.create_bookkeeping()
        dict_label_to_indices = self.meta_alpha_dataset.labels_to_indices
        self.assertEqual(sorted(list(dict_label_to_indices.keys())), self.ds.alphabets)
        for key in dict_label_to_indices:
            self.assertEqual(dict_label_to_indices[key][0], ord(key) - 97)

    def test_union_metadataset(self):
        datasets = [
            self.ds.get_mnist(),
            self.ds.get_omniglot(),
        ]
        datasets = [l2l.data.MetaDataset(ds) for ds in datasets]
        union = l2l.data.UnionMetaDataset(datasets)
        self.assertEqual(len(union), sum([len(ds) for ds in datasets]))
        self.assertTrue(len(union.labels) == sum([len(ds.labels) for ds in datasets]))
        self.assertTrue(len(union.indices_to_labels) == sum([len(ds.indices_to_labels) for ds in datasets]))
        ref = datasets[1][23]
        item = union[len(datasets[0]) + 23]
        # self.assertTrue(item[1] == ref[1])  # Would fail, because labels are remapped.
        self.assertTrue(np.linalg.norm(np.array(item[0]) - np.array(ref[0])) <= 1e-6)
        ref = datasets[1][0]
        item = union[len(datasets[0]) + 0]
        # self.assertTrue(item[1] == ref[1])  # Would fail, because labels are remapped.
        self.assertTrue(np.linalg.norm(np.array(item[0]) - np.array(ref[0])) <= 1e-6)

    def test_filtered_metadataset(self):
        for dataset in [
            self.ds.get_omniglot(),
            self.ds.get_mnist(),
        ]:
            dataset = l2l.data.MetaDataset(dataset)
            all_classes = dataset.labels
            even_classes = [i for i in all_classes if i % 2 == 0]
            odd_classes = [i for i in all_classes if i % 2 == 1]
            evens = l2l.data.FilteredMetaDataset(dataset, even_classes)
            odds = l2l.data.FilteredMetaDataset(dataset, odd_classes)

            self.assertEqual(sorted(even_classes), sorted(evens.labels))
            self.assertEqual(sorted(odd_classes), sorted(odds.labels))

            union = l2l.data.UnionMetaDataset((evens, odds))
            self.assertEqual(sorted(union.labels), sorted(all_classes))

            for label in evens.labels:
                self.assertTrue(label in even_classes)
                self.assertEqual(
                    len(evens.labels_to_indices[label]),
                    len(dataset.labels_to_indices[label])
                )

            for label in odds.labels:
                self.assertTrue(label in odd_classes)
                self.assertEqual(
                    len(odds.labels_to_indices[label]),
                    len(dataset.labels_to_indices[label])
                )


if __name__ == '__main__':
    unittest.main()
