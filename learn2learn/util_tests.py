#!/usr/bin/env python3

import unittest
import torch as th
import learn2learn as l2l 

class UtilTests(unittest.TestCase):

    def setUp(self):
        self.model = th.nn.Sequential(
                th.nn.Linear(4, 64),
                th.nn.ReLU(),
                th.nn.Linear(64, 2)
                )

        self.loss = th.nn.MSELoss()
        self.input = th.tensor[[0],[1],[2],[3]]


    def tearDown(self):
        pass

    def test_module_clone(self):
        x = self.loss(self.model(self.input), th.tensor[[0],[0],[0],[0]]) 
        clone = l2l.clone_module(self.model) 
        assert x.backward() == clone.backward()

    
    def test_module_detach(self):
        pass

    def test_distribution_clone(self):
        pass

    def test_distribution_detach(self):
        pass
        
        
if __name__ == '__main__':
    unittest.main()
