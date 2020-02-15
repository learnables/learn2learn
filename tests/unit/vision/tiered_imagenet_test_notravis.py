#!/usr/bin/env python3

import os
import unittest
import learn2learn as l2l


class TieredImagenetTests(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_download(self):
        root = os.path.expanduser('~/data')
        tiered = l2l.vision.datasets.TieredImagenet(root=root, mode='train', download=True)
        image, label = tiered[12]
        path = os.path.join(root, 'tiered-imagenet', 'train_images_png.pkl')
        self.assertTrue(os.path.exists(path))

        tiered = l2l.vision.datasets.TieredImagenet(root=root, mode='validation', download=True)
        image, label = tiered[12]
        path = os.path.join(root, 'tiered-imagenet', 'val_images_png.pkl')
        self.assertTrue(os.path.exists(path))

        tiered = l2l.vision.datasets.TieredImagenet(root=root, mode='test', download=True)
        image, label = tiered[12]
        path = os.path.join(root, 'tiered-imagenet', 'test_images_png.pkl')
        self.assertTrue(os.path.exists(path))


if __name__ == '__main__':
    unittest.main()
