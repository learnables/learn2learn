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
        self.loss_func = th.nn.MSELoss()
        self.input = th.tensor([[0., 1., 2., 3.]])

    def tearDown(self):
        pass

    def optimizer_step(self, model, gradients):
        for param, gradient in zip(model.parameters(), gradients):
            param.data.sub_(0.01 * gradient)

    def test_module_clone(self):
        original_output = self.model(self.input)
        original_loss = self.loss_func(original_output, th.tensor([[0., 0.]]))
        original_gradients = th.autograd.grad(original_loss,
                                              self.model.parameters(),
                                              retain_graph=True,
                                              create_graph=True)

        cloned_model = l2l.clone_module(self.model)
        self.optimizer_step(self.model, original_gradients)

        cloned_output = cloned_model(self.input)
        cloned_loss = self.loss_func(cloned_output, th.tensor([[0., 0.]]))

        cloned_gradients = th.autograd.grad(cloned_loss,
                                            cloned_model.parameters(),
                                            retain_graph=True,
                                            create_graph=True)

        self.optimizer_step(cloned_model, cloned_gradients)

        for a, b in zip(self.model.parameters(), cloned_model.parameters()):
            assert th.equal(a, b)

    def test_module_detach(self):
        original_output = self.model(self.input)
        original_loss = self.loss_func(original_output, th.tensor([[0., 0.]]))

        original_gradients = th.autograd.grad(original_loss,
                                              self.model.parameters(),
                                              retain_graph=True,
                                              create_graph=True)

        l2l.detach_module(self.model)
        severed = self.model

        self.optimizer_step(self.model, original_gradients)

        severed_output = severed(self.input)
        severed_loss = self.loss_func(severed_output, th.tensor([[0., 0.]]))

        fail = False
        try:
            severed_gradients = th.autograd.grad(severed_loss,
                                                 severed.parameters(),
                                                 retain_graph=True,
                                                 create_graph=True)
        except Exception as e:
            fail = True

        finally:
            assert fail == True

    def test_distribution_clone(self):
        pass

    def test_distribution_detach(self):
        pass


if __name__ == '__main__':
    unittest.main()
