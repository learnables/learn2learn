#!/usr/bin/env python3

import os
import unittest
import learn2learn as l2l


class FC100Tests(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_download(self):
        root = os.path.expanduser('./data')
        fc100 = l2l.vision.datasets.FC100(root=root,
                                          mode='train',
                                          download=True)
        image, label = fc100[12]
        path = os.path.join(root, 'FC100_train.pickle')
        self.assertTrue(os.path.exists(path))

        fc100 = l2l.vision.datasets.FC100(root=root,
                                          mode='validation',
                                          download=True)
        image, label = fc100[12]
        path = os.path.join(root, 'FC100_val.pickle')
        self.assertTrue(os.path.exists(path))

        fc100 = l2l.vision.datasets.FC100(root=root,
                                          mode='test',
                                          download=True)
        image, label = fc100[12]
        path = os.path.join(root, 'FC100_test.pickle')
        self.assertTrue(os.path.exists(path))


if __name__ == '__main__':
    unittest.main()
