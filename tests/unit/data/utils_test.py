#!/usr/bin/env python3

import unittest
import torch
import learn2learn as l2l


class DataUtilsTests(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_partition_task(self):
        TOTAL_SAMPLES = 20
        data = torch.randn(TOTAL_SAMPLES, 10)
        labels = torch.randn(TOTAL_SAMPLES, 1)
        for shots in range(TOTAL_SAMPLES):
            queries = TOTAL_SAMPLES - shots
            (S_X, S_Y), (Q_X, Q_Y) = l2l.data.partition_task(
                data,
                labels,
                shots,
            )
            self.assertAlmostEquals(shots, S_X.size(0))
            self.assertAlmostEquals(shots, S_Y.size(0))
            self.assertAlmostEquals(queries, Q_X.size(0))
            self.assertAlmostEquals(queries, Q_Y.size(0))


if __name__ == "__main__":
    unittest.main()
