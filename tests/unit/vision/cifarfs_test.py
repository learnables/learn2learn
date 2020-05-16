#!/usr/bin/env python3

import os
import unittest
import learn2learn as l2l


class CIFARFSTests(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_download(self):
        cifarfs = l2l.vision.datasets.CIFARFS(root='./data', download=True)
        path = os.path.join('./data', 'cifarfs', 'processed')
        self.assertTrue(os.path.exists(path))


if __name__ == '__main__':
    unittest.main()
