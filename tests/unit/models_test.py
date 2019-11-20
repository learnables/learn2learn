#!/usr/bin/env python3

import unittest

import learn2learn as l2l
import torch as th

N = 5
M = 3
EPSILON = 1e-5


def vec(mat):
    return mat.t().contiguous().view(-1, 1)


def kronecker_product(t1, t2):
    """
    Computes the Kronecker product between two tensors.
    See https://en.wikipedia.org/wiki/Kronecker_product
    and https://discuss.pytorch.org/t/kronecker-product/3919/4
    """
    t1_height, t1_width = t1.size()
    t2_height, t2_width = t2.size()
    out_height = t1_height * t2_height
    out_width = t1_width * t2_width

    tiled_t2 = t2.repeat(t1_height, t1_width)
    expanded_t1 = (
        t1.unsqueeze(2)
          .unsqueeze(3)
          .repeat(1, t2_height, t2_width, 1)
          .view(out_height, out_width)
    )

    return expanded_t1 * tiled_t2


def close(x, y):
    return (x - y).norm(p=2) <= EPSILON


class TestModels(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_kronecker_addmm(self):
        for _ in range(10):
            mat1 = th.randn(N, N)
            mat2 = th.randn(M, M)
            mat3 = th.randn(N, M)
            bias = th.randn(N, M)
            result = vec(l2l.models.kronecker_addmm(mat1, mat2, mat3, bias))
            kron = kronecker_product(mat2.t(), mat1)
            ref = kron @ vec(mat3) + vec(bias)
            self.assertTrue(close(ref, result))

    def test_cholesky_addmm(self):
        for rank in range(10):
            mat1 = th.randn(N, rank)
            mat2 = th.randn(N, N)
            bias = th.randn(N, N)
            result = l2l.models.cholesky_addmm(mat1, mat2, bias)
            ref = th.chain_matmul(mat1, mat1.t(), mat2) + bias
            self.assertTrue(close(ref, result))


if __name__ == '__main__':
    unittest.main()
