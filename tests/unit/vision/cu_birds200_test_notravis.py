#!/usr/bin/env python3

import os
import unittest
import learn2learn as l2l


class BirdsTests(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_download(self):
        root = os.path.expanduser('~/data')
        birds = l2l.vision.datasets.CUBirds200(
            root=root,
            mode='train',
            download=True,
        )
        image, label = birds[12]
        path = os.path.join(root, 'cubirds200', 'CUB_200_2011', 'images')
        self.assertTrue(os.path.exists(path))

        birds = l2l.vision.datasets.CUBirds200(
            root=root,
            mode='validation',
        )
        image, label = birds[12]

        birds = l2l.vision.datasets.CUBirds200(root=root, mode='test')
        image, label = birds[12]


if __name__ == '__main__':
    unittest.main()
