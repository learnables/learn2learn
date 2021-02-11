#!/usr/bin/env python3

import unittest
import torch
import learn2learn as l2l


def kronecker_product(A, B):
    return torch.einsum("ab,cd->acbd", A, B).view(
        A.size(0) * B.size(0), A.size(1) * B.size(1)
    )


def vec(A):
    return A.t().flatten() if len(A.shape) > 1 else A


class KroneckerLinearTests(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def _get_l2l_and_naive(self, m, n, vectorize_input):
        x = torch.randn(n, m) if not vectorize_input else torch.randn(m * n)
        kronecker = l2l.nn.KroneckerLinear(n, m)
        l2l_result = kronecker(x)

        K = kronecker_product(kronecker.left, kronecker.right)
        bias = vec(kronecker.bias)
        x_vec = vec(x)  # Make sure input is a vector for naive approach
        naive_result = K @ x_vec + bias  # Naive result
        return x, l2l_result, naive_result

    def _test_template(self, m, n):
        input_matrix, l2l_res, naive_res = self._get_l2l_and_naive(m, n, False)
        self.assertTrue(input_matrix.shape == l2l_res.shape)
        l2l_res = vec(l2l_res)
        self.assertTrue(l2l_res.shape == naive_res.shape)
        self.assertTrue(torch.all(torch.eq(l2l_res, naive_res)))

    def _test_template_1d(self, m, n):
        input_vec, l2l_res, naive_res = self._get_l2l_and_naive(m, n, True)
        self.assertTrue(input_vec.shape == l2l_res.shape)
        self.assertTrue(l2l_res.shape == naive_res.shape)
        self.assertTrue(torch.all(torch.eq(l2l_res, naive_res)))

    def test_simple(self):
        m, n = 3, 2
        self._test_template(m, n)

    def test_simple_1d(self):
        m, n = 3, 2
        self._test_template_1d(m, n)

    def test_n_edge(self):
        m, n = 4, 1
        self._test_template(m, n)

    def test_n_edge_1d(self):
        m, n = 4, 1
        self._test_template_1d(m, n)

    def test_m_edge(self):
        m, n = 1, 5
        self._test_template(m, n)

    def test_m_edge_1d(self):
        m, n = 1, 5
        self._test_template_1d(m, n)

    def test_m_n_edge(self):
        m, n = 1, 1
        self._test_template(m, n)

    def test_m_n_edge_1d(self):
        m, n = 1, 1
        self._test_template_1d(m, n)

    def test_illegal_dimensions(self):
        m, n = 2, 3
        x = torch.randn(n, m + 1)
        kronecker = l2l.nn.KroneckerLinear(n, m)
        self.assertRaises(ValueError, kronecker, x)

    def test_illegal_dimensions_1d(self):
        m, n = 2, 3
        x = torch.randn(n * (m + 1))
        kronecker = l2l.nn.KroneckerLinear(n, m)
        self.assertRaises(ValueError, kronecker, x)


if __name__ == "__main__":
    unittest.main()
