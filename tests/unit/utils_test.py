#!/usr/bin/env python3

import unittest
import copy
import torch
import learn2learn as l2l

EPSILON = 1e-8


def ref_clone_module(module):
    """
    Note: This implementation does not work for RNNs.
    It requires calling learner.rnn._apply(lambda x: x) before
    each forward call.
    See this issue for more details:
    https://github.com/learnables/learn2learn/issues/139

    Note: This implementation also does not work for Modules that re-use
    parameters from another Module.
    See this issue for more details:
    https://github.com/learnables/learn2learn/issues/174
    """
    # First, create a copy of the module.
    clone = copy.deepcopy(module)

    # Second, re-write all parameters
    if hasattr(clone, '_parameters'):
        for param_key in module._parameters:
            if module._parameters[param_key] is not None:
                cloned = module._parameters[param_key].clone()
                clone._parameters[param_key] = cloned

    # Third, handle the buffers if necessary
    if hasattr(clone, '_buffers'):
        for buffer_key in module._buffers:
            if clone._buffers[buffer_key] is not None and \
                    clone._buffers[buffer_key].requires_grad:
                clone._buffers[buffer_key] = module._buffers[buffer_key].clone()

    # Then, recurse for each submodule
    if hasattr(clone, '_modules'):
        for module_key in clone._modules:
            clone._modules[module_key] = ref_clone_module(module._modules[module_key])
    return clone


class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(4, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.model(x)


class UtilTests(unittest.TestCase):

    def setUp(self):
        self.model = Model()
        self.loss_func = torch.nn.MSELoss()
        self.input = torch.tensor([[0., 1., 2., 3.]])

    def tearDown(self):
        pass

    def optimizer_step(self, model, gradients):
        for param, gradient in zip(model.parameters(), gradients):
            param.data.sub_(0.01 * gradient)

    def test_clone_module_basics(self):
        original_output = self.model(self.input)
        original_loss = self.loss_func(original_output, torch.tensor([[0., 0.]]))
        original_gradients = torch.autograd.grad(original_loss,
                                                 self.model.parameters(),
                                                 retain_graph=True,
                                                 create_graph=True)

        cloned_model = l2l.clone_module(self.model)
        self.optimizer_step(self.model, original_gradients)

        cloned_output = cloned_model(self.input)
        cloned_loss = self.loss_func(cloned_output, torch.tensor([[0., 0.]]))

        cloned_gradients = torch.autograd.grad(cloned_loss,
                                               cloned_model.parameters(),
                                               retain_graph=True,
                                               create_graph=True)

        self.optimizer_step(cloned_model, cloned_gradients)

        for a, b in zip(self.model.parameters(), cloned_model.parameters()):
            self.assertTrue(torch.equal(a, b))

    def test_clone_module_nomodule(self):
        # Tests that we can clone non-module objects
        class TrickyModule(torch.nn.Module):

            def __init__(self):
                super(TrickyModule, self).__init__()
                self.tricky_modules = torch.nn.ModuleList([
                    torch.nn.Linear(2, 1),
                    None,
                    torch.nn.Linear(1, 1),
                ])

        model = TrickyModule()
        clone = l2l.clone_module(model)
        for i, submodule in enumerate(clone.tricky_modules):
            if i % 2 == 0:
                self.assertTrue(submodule is not None)
            else:
                self.assertTrue(submodule is None)

    def test_clone_module_models(self):
        ref_models = [l2l.vision.models.OmniglotCNN(10),
                  l2l.vision.models.MiniImagenetCNN(10)]
        l2l_models = [copy.deepcopy(m) for m in ref_models]
        inputs = [torch.randn(5, 1, 28, 28), torch.randn(5, 3, 84, 84)]


        # Compute reference gradients
        ref_grads = []
        for model, X in zip(ref_models, inputs):
            for iteration in range(10):
                model.zero_grad()
                clone = ref_clone_module(model)
                out = clone(X)
                out.norm(p=2).backward()
                self.optimizer_step(model, [p.grad for p in model.parameters()])
                ref_grads.append([p.grad.clone().detach() for p in model.parameters()])

        # Compute cloned gradients
        l2l_grads = []
        for model, X in zip(l2l_models, inputs):
            for iteration in range(10):
                model.zero_grad()
                clone = l2l.clone_module(model)
                out = clone(X)
                out.norm(p=2).backward()
                self.optimizer_step(model, [p.grad for p in model.parameters()])
                l2l_grads.append([p.grad.clone().detach() for p in model.parameters()])

        # Compare gradients and model parameters
        for ref_g, l2l_g in zip(ref_grads, l2l_grads):
            for r_g, l_g in zip(ref_g, l2l_g):
                self.assertTrue(torch.equal(r_g, l_g))
        for ref_model, l2l_model in zip(ref_models, l2l_models):
            for ref_p, l2l_p in zip(ref_model.parameters(), l2l_model.parameters()):
                self.assertTrue(torch.equal(ref_p, l2l_p))

    def test_rnn_clone(self):
        # Tests: https://github.com/learnables/learn2learn/issues/139
        # The test is mainly about whether we can clone and adapt RNNs.
        # See issue for details.
        N_STEPS = 3
        for rnn_class in [
            torch.nn.RNN,
            torch.nn.LSTM,
            torch.nn.GRU,
        ]:
            torch.manual_seed(1234)
            model = rnn_class(2, 1)
            maml = l2l.algorithms.MAML(model, lr=1e-3, allow_unused=False)
            optim = torch.optim.SGD(maml.parameters(), lr=0.001)
            data = torch.randn(30, 500, 2)

            # Adapt and measure loss
            learner = maml.clone()
            for step in range(N_STEPS):
                pred, hidden = learner(data)
                loss = pred.norm(p=2)
                learner.adapt(loss)
            pred, _ = learner(data)
            first_loss = pred.norm(p=2)

            # Take an optimization step
            optim.zero_grad()
            first_loss.backward()
            optim.step()
            first_loss = first_loss.item()

            # Adapt a second time
            learner = maml.clone()
            for step in range(N_STEPS):
                pred, hidden = learner(data)
                loss = pred.norm(p=2)
                learner.adapt(loss)
            pred, _ = learner(data)
            second_loss = pred.norm(p=2)
            second_loss = second_loss.item()

            # Ensure we did better
            self.assertTrue(first_loss > second_loss)

    def test_module_clone_shared_params(self):
        # Tests proper use of memo parameter

        class TestModule(torch.nn.Module):
            def __init__(self):
                super(TestModule, self).__init__()
                cnn = [
                    torch.nn.Conv2d(3, 32, 3, 2, 1),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(32, 32, 3, 2, 1),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(32, 32, 3, 2, 1),
                    torch.nn.ReLU(),
                ]
                self.seq = torch.nn.Sequential(*cnn)
                self.head = torch.nn.Sequential(*[
                    torch.nn.Conv2d(32, 32, 3, 2, 1),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(32, 100, 3, 2, 1)]
                )
                self.net = torch.nn.Sequential(self.seq, self.head)

            def forward(self, x):
                return self.net(x)

        original = TestModule()
        clone = l2l.clone_module(original)
        self.assertTrue(
            len(list(clone.parameters())) == len(list(original.parameters())),
            'clone and original do not have same number of parameters.',
        )

        orig_params = [p.data_ptr() for p in original.parameters()]
        duplicates = [p.data_ptr() in orig_params for p in clone.parameters()]
        self.assertTrue(not any(duplicates), 'clone() forgot some parameters.')

    def test_module_update_shared_params(self):
        # Tests proper use of memo parameter

        class TestModule(torch.nn.Module):
            def __init__(self):
                super(TestModule, self).__init__()
                cnn = [
                    torch.nn.Conv2d(3, 32, 3, 2, 1),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(32, 32, 3, 2, 1),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(32, 32, 3, 2, 1),
                    torch.nn.ReLU(),
                ]
                self.seq = torch.nn.Sequential(*cnn)
                self.head = torch.nn.Sequential(*[
                    torch.nn.Conv2d(32, 32, 3, 2, 1),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(32, 100, 3, 2, 1)]
                )
                self.net = torch.nn.Sequential(self.seq, self.head)

            def forward(self, x):
                return self.net(x)

        original = TestModule()
        num_original = len(list(original.parameters()))
        clone = l2l.clone_module(original)
        updates = [torch.randn_like(p) for p in clone.parameters()]
        l2l.update_module(clone, updates)
        num_clone = len(list(clone.parameters()))
        self.assertTrue(
            num_original == num_clone,
            'clone and original do not have same number of parameters.',
        )
        for p, c, u in zip(original.parameters(), clone.parameters(), updates):
            self.assertTrue(torch.norm(p + u - c, p=2) <= EPSILON, 'clone is not original + update.')

        orig_params = [p.data_ptr() for p in original.parameters()]
        duplicates = [p.data_ptr() in orig_params for p in clone.parameters()]
        self.assertTrue(not any(duplicates), 'clone() forgot some parameters.')

    def test_module_detach(self):
        original_output = self.model(self.input)
        original_loss = self.loss_func(
            original_output,
            torch.tensor([[0., 0.]])
        )

        original_gradients = torch.autograd.grad(original_loss,
                                                 self.model.parameters(),
                                                 retain_graph=True,
                                                 create_graph=True)

        l2l.detach_module(self.model)
        severed = self.model

        self.optimizer_step(self.model, original_gradients)

        severed_output = severed(self.input)
        severed_loss = self.loss_func(severed_output, torch.tensor([[0., 0.]]))

        fail = False
        try:
            severed_gradients = torch.autograd.grad(severed_loss,
                                                    severed.parameters(),
                                                    retain_graph=True,
                                                    create_graph=True)
        except Exception as e:
            fail = True

        finally:
            assert fail == True

    def test_module_detach_keep_requires_grad(self):
        l2l.detach_module(self.model, keep_requires_grad=True)
        self.assertTrue(all(p.requires_grad for p in self.model.parameters()))
        l2l.detach_module(self.model)
        self.assertTrue(all(not p.requires_grad for p in self.model.parameters()))

    def test_distribution_clone(self):
        pass

    def test_distribution_detach(self):
        pass


if __name__ == '__main__':
    unittest.main()
