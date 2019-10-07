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

    def test_tg_with_none_task(self):
        """ In this test we want to ensure that we sample from all possible tasks i.e 20 given 1000 iterations.
        """
        ways = 2
        # tg1 allows us to check if next() works
        tg1 = TaskGenerator(self.ds.tensor_dataset, tasks=None, ways=ways)

        # tg2 allows us to check if .sample() works
        tg2 = TaskGenerator(self.ds.tensor_dataset, tasks=None, ways=ways)

        # tg3 allows us to check that if we loop less than 20 times i.e permutations possible
        # then we don't generate all the possible tasks
        tg3 = TaskGenerator(self.ds.tensor_dataset, tasks=None, ways=ways)

        permutations_possible = {(0, 1), (0, 2), (0, 3), (0, 4),
                                 (1, 0), (1, 2), (1, 3), (1, 4),
                                 (2, 0), (2, 1), (2, 3), (2, 4),
                                 (3, 0), (3, 1), (3, 2), (3, 4),
                                 (4, 0), (4, 1), (4, 2), (4, 3)}

        tasks_generated_1 = set()
        tasks_generated_2 = set()
        tasks_generated_3 = set()
        for idx in range(1000):
            task1 = next(tg1)
            task2 = tg2.sample()

            tasks_generated_1.add(tuple(task1.sampled_task))
            tasks_generated_2.add(tuple(task2.sampled_task))
            if idx < 15:
                task3 = next(tg3)
                tasks_generated_3.add(tuple(task3.sampled_task))
            if idx == 200:
                break

        self.assertEqual(len(permutations_possible - tasks_generated_1), 0, "TG1 didn't generate all the possible tasks.")
        self.assertEqual(len(permutations_possible - tasks_generated_2), 0, "TG2 didn't generate all the possible tasks.")
        self.assertGreater(len(permutations_possible - tasks_generated_3), 1, "TG3, generated all the possible tasks")
        self.assertEqual(len(tg1), 0, "Len of TG1 isn't as expected.")

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
