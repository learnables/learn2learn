#!/usr/bin/env python3

import unittest
import learn2learn as l2l


class UtilTests(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_tasksets(self):
        names = l2l.vision.benchmarks.list_tasksets()
        for name in names:
            tasks = l2l.vision.benchmarks.get_tasksets(name)
            self.assertTrue(hasattr(tasks, 'train'))
            self.assertTrue(hasattr(tasks, 'validation'))
            self.assertTrue(hasattr(tasks, 'test'))


if __name__ == '__main__':
    unittest.main()
