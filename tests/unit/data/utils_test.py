#!/usr/bin/env python3

import unittest
import torch
import learn2learn as l2l

EPS = 1e-8


def close(a, b):
    return torch.norm(a - b).item() <= EPS


class DataUtilsTests(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_partition_task(self):
        NUM_SAMPLES = 20
        data = torch.randn(NUM_SAMPLES, 10)
        labels = torch.randint(low=0, high=1, size=(NUM_SAMPLES, ))
        for shots in range(NUM_SAMPLES):
            queries = NUM_SAMPLES - shots
            (S_X, S_Y), (Q_X, Q_Y) = l2l.data.partition_task(
                data,
                labels,
                shots,
            )
            self.assertAlmostEquals(shots, S_X.size(0))
            self.assertAlmostEquals(shots, S_Y.size(0))
            self.assertAlmostEquals(queries, Q_X.size(0))
            self.assertAlmostEquals(queries, Q_Y.size(0))

    def test_infinite_iterator(self):
        NUM_SAMPLES = 20
        data = torch.randn(NUM_SAMPLES, 10)
        dataset = torch.utils.data.TensorDataset(data)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            shuffle=False,
            batch_size=1,
        )
        inf_dataloader = l2l.data.InfiniteIterator(dataloader)
        counter = 0
        for x in inf_dataloader:
            self.assertTrue(close(x[0], data[counter % NUM_SAMPLES]))
            if counter > NUM_SAMPLES * 5:
                break
            counter += 1


if __name__ == "__main__":
    unittest.main()
