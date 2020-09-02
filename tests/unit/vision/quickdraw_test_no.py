#!/usr/bin/env python3

import os
import unittest
import learn2learn as l2l


class QuickdrawTests(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_download(self):
        root = os.path.expanduser('~/data')
        quickdraw = l2l.vision.datasets.Quickdraw(
            root=root,
            mode='train',
            download=True,
        )
        image, label = quickdraw[12]
        path = os.path.join(root, 'quickdraw')
        self.assertTrue(os.path.exists(path))

        quickdraw = l2l.vision.datasets.Quickdraw(
            root=root,
            mode='validation',
        )
        image, label = quickdraw[12]

        quickdraw = l2l.vision.datasets.Quickdraw(root=root, mode='test')
        image, label = quickdraw[12]


if __name__ == '__main__':
    unittest.main()
