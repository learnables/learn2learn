#!/usr/bin/env python3

import unittest
import torch
import learn2learn as l2l


def close(x, y):
    return (x - y).norm(p=2) <= 1e-8


class NNMiscTests(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def _get_scale_shape(self, shape):
        if isinstance(shape, int):
            shape = (shape, )
        return shape

    def test_scale(self):
        for shape in [1, 4, (1, ), [2, ], [3, 2], (1, 2, 3)]:
            for alpha in [-1, 0, 1, 0.5]:
                scale = l2l.nn.Scale(shape=shape, alpha=alpha)
                x = torch.ones(
                    size=self._get_scale_shape(shape),
                )
                self.assertTrue(x.size() == scale.alpha.size())
                self.assertTrue(close(alpha * x, scale(x)))


if __name__ == "__main__":
    unittest.main()
