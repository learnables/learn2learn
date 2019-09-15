import random
import unittest
from copy import deepcopy
from unittest import TestCase

import numpy as np

from learn2learn.data import MetaDataset, TaskGenerator
from .util_datasets import TestDatasets


class TestTaskGenerator(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ds = TestDatasets()
        cls.meta_tensor_dataset = MetaDataset(cls.ds.tensor_dataset)
        cls.meta_str_dataset = MetaDataset(cls.ds.str_dataset)
        cls.meta_alpha_dataset = MetaDataset(cls.ds.alphabet_dataset)

    def test_does_not_fail_with_torch_dataset(self):
        try:
            TaskGenerator(self.ds.tensor_dataset)
            return True
        except:
            return False

    def test_tg_with_list_task(self):
        num_tasks = 20
        ways = 3
        classes = self.ds.tensor_classes
        tasks = [random.sample(classes, k=ways) for i in range(num_tasks)]
        tg = TaskGenerator(self.ds.tensor_dataset, tasks=tasks, ways=ways)
        self.assertEqual(tg.tasks, tasks)

    def test_tg_with_int_task(self):
        num_tasks = 20
        ways = 3
        tg = TaskGenerator(self.ds.tensor_dataset, tasks=num_tasks, ways=ways)
        self.assertEqual(len(tg), num_tasks)

    def test_tg_fails_with_np_array(self):
        num_tasks = 20
        ways = 3
        tasks = np.random.randint(low=0, high=max(self.ds.tensor_classes), size=(num_tasks, ways))

        try:
            tg = TaskGenerator(self.ds.tensor_dataset, tasks=tasks, ways=ways)
            return False
        except TypeError:
            return True
        finally:
            return False

    def test_tasks_dont_change(self):
        num_tasks = 20
        ways = 3
        classes = self.ds.tensor_classes
        tasks = [random.sample(classes, k=ways) for i in range(num_tasks)]

        tg = TaskGenerator(self.ds.tensor_dataset, tasks=tasks, ways=ways)
        tg_tasks = deepcopy(tg.tasks)

        self.assertEqual(tasks, tg_tasks)

        _ = tg.sample()
        self.assertEqual(tasks, tg.tasks)

        _ = next(tg)
        self.assertEqual(tasks, tg.tasks)

    def test_next_tg(self):
        num_tasks = 20
        ways = 3
        classes = self.ds.tensor_classes
        tasks = [random.sample(classes, k=ways) for i in range(num_tasks)]
        tg = TaskGenerator(self.ds.tensor_dataset, tasks=tasks, ways=ways)
        for i, task in enumerate(tg):
            self.assertEqual(task.sampled_task, tasks[i])


if __name__ == '__main__':
    unittest.main()
