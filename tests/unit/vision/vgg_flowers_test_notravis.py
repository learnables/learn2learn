#!/usr/bin/env python3

import os
import unittest
import learn2learn as l2l


class FlowersTests(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_download(self):
        root = os.path.expanduser('~/data')
        flowers = l2l.vision.datasets.VGGFlower102(root=root, mode='train', download=True)
        image, label = flowers[12]
        path = os.path.join(root, 'vgg_flower102', 'imagelabels.mat')
        self.assertTrue(os.path.exists(path))

        flowers = l2l.vision.datasets.VGGFlower102(root=root, mode='validation')
        image, label = flowers[12]

        flowers = l2l.vision.datasets.VGGFlower102(root=root, mode='test')
        image, label = flowers[12]


if __name__ == '__main__':
    unittest.main()
