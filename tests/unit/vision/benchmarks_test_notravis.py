#!/usr/bin/env python3

import unittest
import learn2learn as l2l

TOO_BIG_TO_TEST = [
    'tiered-imagenet',
]


class UtilTests(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_tasksets(self):
        names = l2l.vision.benchmarks.list_tasksets()
        for name in names:
            if name in TOO_BIG_TO_TEST:
                continue
            tasksets = l2l.vision.benchmarks.get_tasksets(name, root='~/data')
            self.assertTrue(hasattr(tasksets, 'train'))
            batch = tasksets.train.sample()
            self.assertTrue(batch is not None)
            self.assertTrue(hasattr(tasksets, 'validation'))
            batch = tasksets.validation.sample()
            self.assertTrue(batch is not None)
            self.assertTrue(hasattr(tasksets, 'test'))
            batch = tasksets.test.sample()
            self.assertTrue(batch is not None)
            del tasksets, batch


if __name__ == '__main__':
    unittest.main()
