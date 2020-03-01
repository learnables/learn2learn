#!/usr/bin/env python3

import os
import unittest
import learn2learn as l2l


class AircraftTests(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_download(self):
        root = os.path.expanduser('~/data')
        aircraft = l2l.vision.datasets.FGVCAircraft(root=root, mode='train', download=True)
        image, label = aircraft[12]
        path = os.path.join(root, 'fgvc_aircraft')
        self.assertTrue(os.path.exists(path))

        aircraft = l2l.vision.datasets.FGVCAircraft(root=root, mode='validation')
        image, label = aircraft[12]
        path = os.path.join(root, 'fgvc_aircraft')
        self.assertTrue(os.path.exists(path))

        aircraft = l2l.vision.datasets.FGVCAircraft(root=root, mode='test')
        image, label = aircraft[12]
        path = os.path.join(root, 'fgvc_aircraft')
        self.assertTrue(os.path.exists(path))


if __name__ == '__main__':
    unittest.main()
