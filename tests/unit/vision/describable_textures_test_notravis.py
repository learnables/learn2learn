#!/usr/bin/env python3

import os
import unittest
import learn2learn as l2l


class DTDTests(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_download(self):
        root = os.path.expanduser('~/data')
        dtd = l2l.vision.datasets.DescribableTextures(
            root=root,
            mode='train',
            download=True,
        )
        image, label = dtd[12]
        path = os.path.join(
            root,
            'describable_textures',
            'dtd',
            'images',
        )
        self.assertTrue(os.path.exists(path))

        dtd = l2l.vision.datasets.DescribableTextures(
            root=root,
            mode='validation',
        )
        image, label = dtd[12]

        dtd = l2l.vision.datasets.DescribableTextures(root=root, mode='test')
        image, label = dtd[12]


if __name__ == '__main__':
    unittest.main()
