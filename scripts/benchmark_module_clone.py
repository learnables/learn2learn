#!/usr/bin/env python3

import torch
import learn2learn as l2l
import ppt
import time

GPU = True
N = 100


def clone_module(module):
    # First, create a copy of the module.
    # From: https://github.com/pytorch/pytorch/blob/65bad41cbec096aa767b3752843eddebf845726f/torch/nn/modules/module.py#L1171
    clone = module.__new__(type(module))
    clone.__dict__ = module.__dict__.copy()
    clone._parameters = clone._parameters.copy()
    clone._buffers = clone._buffers.copy()
    clone._modules = clone._modules.copy()

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
            clone._modules[module_key] = clone_module(module._modules[module_key])
    return clone


model = l2l.vision.models.MiniImagenetCNN(10)
if GPU:
    model.to('cuda')

clone_start = time.time()
ppt.time('clone')
for _ in range(N):
    clone = l2l.clone_module(model)
ppt.stop()
clone_end = time.time()


fast_start = time.time()
ppt.time('fast')
for _ in range(N):
    clone = clone_module(model)
ppt.stop()
fast_end = time.time()

ppt.summary()
print('user', 'clone', clone_end - clone_start)
print('user', 'fast', fast_end - fast_start)


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

model = Model()
model.zero_grad()

def optimizer_step(model, gradients):
    for param, gradient in zip(model.parameters(), gradients):
        param.data.sub_(0.01 * gradient)


input = torch.tensor([[0., 1., 2., 3.]])
original_output = model(input)
original_loss = torch.nn.functional.mse_loss(original_output, torch.tensor([[0., 0.]]))
original_gradients = torch.autograd.grad(original_loss,
                                         model.parameters(),
                                         retain_graph=True,
                                         create_graph=True)

cloned_model = clone_module(model)
optimizer_step(model, original_gradients)

cloned_output = cloned_model(input)
cloned_loss = torch.nn.functional.mse_loss(cloned_output, torch.tensor([[0., 0.]]))

cloned_gradients = torch.autograd.grad(cloned_loss,
                                       cloned_model.parameters(),
                                       retain_graph=True,
                                       create_graph=True)

optimizer_step(cloned_model, cloned_gradients)

for a, b in zip(model.parameters(), cloned_model.parameters()):
    assert torch.equal(a, b)

