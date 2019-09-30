#!/usr/bin/env python3

import os
import unittest
import learn2learn as l2l


class UtilTests(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_download(self):
        cifarfs = l2l.vision.datasets.CIFARFS(root='./data')
        path = os.path.join('./data', 'cifarfs', 'preprocessed')
        self.assertTrue(os.path.exists(path))
