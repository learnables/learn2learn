#!/usr/bin/env python3

import unittest
import torch as th
import learn2learn as l2l 


class Model(th.nn.Module):

    def __init__(self):
        super().__init__()
        self.model = th.nn.Sequential(
                          th.nn.Linear(4, 64),
                          th.nn.Tanh(),
                          th.nn.Linear(64, 2)
                          )

    def forward(self, x):
        return self.model(x)


class UtilTests(unittest.TestCase):

    def setUp(self):
        self.model = Model()
        self.loss = th.nn.MSELoss()
        self.input = th.tensor([[0.,1.,2.,3.]])


    def tearDown(self):
        pass

    def test_module_clone(self):
        x = self.model(self.input)
        y = self.loss(x, th.tensor([[0.,0.]]))

        gradients = th.autograd.grad(y,
                         self.model.parameters(),
                         retain_graph=True,
                         create_graph=True)

        clone = l2l.clone_module(self.model) 

        for param, gradient in zip(self.model.parameters(), gradients):
            param = param - .01 * gradient

        x = clone(self.input)
        y = self.loss(x, th.tensor([[0.,0.]]))

        gradients = th.autograd.grad(y,
                         clone.parameters(),
                         retain_graph=True,
                         create_graph=True)

        for param, gradient in zip(clone.parameters(), gradients):
            param = param - .01 * gradient

        for a,b in zip(self.model.parameters(), clone.parameters()):
            assert th.equal(a,b)

    
    def test_module_detach(self):
        x = self.model(self.input)
        y = self.loss(x, th.tensor([[0.,0.]]))

        gradients = th.autograd.grad(y,
                         self.model.parameters(),
                         retain_graph=True,
                         create_graph=True)

        l2l.detach_module(self.model) 
        severed = self.model

        for param, gradient in zip(self.model.parameters(), gradients):
            param = param - .01 * gradient

        x = severed(self.input)
        y = self.loss(x, th.tensor([[0.,0.]]))

        fail = False
        try:
            gradients = th.autograd.grad(y,
                                         severed.parameters(),
                                         retain_graph=True,
                                         create_graph=True)
        except:
            fail = True

        finally:
            assert fail == True

    def test_distribution_clone(self):
        pass

    def test_distribution_detach(self):
        pass
        
        
if __name__ == '__main__':
    unittest.main()
