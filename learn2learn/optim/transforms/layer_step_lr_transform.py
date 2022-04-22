# -*- coding: utf-8 -*-
# vim:fenc=utf-8

"""
Per-Layer and Per-Layer Per-Step Learning Rate transforms for the GBML algorithm.
"""

import learn2learn as l2l
import numpy as np
import random
import torch


class PerStepLR(torch.nn.Module):
    def __init__(self, init_lr: float, steps: int):
        super().__init__()
        self.lrs = torch.nn.Parameter(
            data=torch.ones(steps) * init_lr,
            requires_grad=True,
        )
        self._current_step = 0

    def forward(self, grad):
        updates = -self.lrs[self._current_step] * grad
        self._current_step += 1
        self._current_step = min(
            len(self.lrs) - 1, self._current_step
        )  # avoids overflow
        return updates

    def __str__(self):
        return str(self.lrs)


class PerLayerPerStepLRTransform:
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/optim/transforms/layer_step_lr_transform.py)

    **Description**

    The PerLayerPerStepLRTransform creates a per-step transform for each layer of a given module.

    This can be used with the GBML algorithm to reproduce the *LSLR* improvement of MAML++ proposed by
    Antoniou et al.

    **Arguments**

    * **init_lr** (float) - The initial learning rate for each adaptation step and layer.
    * **steps** (int) - The number of adaptation steps.
    * **model** (torch.nn.Module) - The module being updated with the learning rates. This is
        needed to define the learning rates for each layer.

    **Example**
    ~~~python
    model = torch.nn.Sequential(
        torch.nn.Linear(128, 24), torch.nn.Linear(24, 16), torch.nn.Linear(16, 10)
    )
    transform = PerLayerPerStepLRTransform(1e-3, N_STEPS, model)
    metamodel = l2l.algorithms.GBML(
        model,
        transform,
        allow_nograd=True,
        lr=0.001,
        adapt_transform=False,
        pass_param_names=True,  # This is needed for this transform to find the module's layers
    )
    opt = torch.optim.Adam(metamodel.parameters(), lr=1.0)
    ~~~
    """

    def __init__(self, init_lr, steps, model):
        self._lslr = {}
        for layer_name, layer in model.named_modules():
            # If the layer has learnable parameters
            if (
                len(
                    [
                        name
                        for name, param in layer.named_parameters(recurse=False)
                        if param.requires_grad
                    ]
                )
                > 0
            ):
                # lslr[layer_name.replace("module.", "").replace(".", "-")] = torch.nn.Parameter(
                self._lslr[layer_name] = PerStepLR(init_lr, steps)

    def __call__(self, name, param):
        name = name[
            : name.rfind(".")
        ]  # Extract the layer name from the named parameter
        assert name in self._lslr, "No matching LR found for layer."
        return self._lslr[name]

    def __str__(self):
        string = ""
        for layer, lslr in self._lslr.items():
            string += f"Layer {layer}: {lslr}\n"
        return string


class PerStepLRTransform:
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/optim/transforms/layer_step_lr_transform.py)

    **Description**

    The PerStepLRTransform creates a per-step transform for inner-loop-based algorithms.

    This can be used with the GBML algorithm to reproduce the *LSLR* improvement of MAML++ proposed by
    Antoniou et al, with the same learning rates for all layers.

    **Arguments**

    * **init_lr** (float) - The initial learning rate for each adaptation step.
    * **steps** (int) - The number of adaptation steps.

    **Example**
    ~~~python
    model = torch.nn.Linear(128, 10)
    transform = PerStepLRTransform(1e-3, N_STEPS)
    metamodel = l2l.algorithms.GBML(
        model,
        transform,
        allow_nograd=True,
        lr=0.001,
        adapt_transform=False,
    )
    opt = torch.optim.Adam(metamodel.parameters(), lr=1.0)
    ~~~
    """

    def __init__(self, init_lr, steps):
        self._obj = PerStepLR(init_lr, steps)

    def __call__(self, param):
        return self._obj

    def __str__(self):
        return str(self._obj)

    def parameters(self):
        return self._obj.parameters()


if __name__ == "__main__":

    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)

    N_DIMS = 32
    N_SAMPLES = 128
    N_STEPS = 5
    device = torch.device("cpu")

    print("[*] Testing per-step LR with one linear layer")
    model = torch.nn.Linear(N_DIMS, 10)
    transform = PerStepLRTransform(1e-3, N_STEPS)
    # Setting adapt_transform=True means that the transform will be updated in
    # the *adapt* function, which is not what we want. We want it to compute gradients during
    # eval_loss.backward() only, so that it's updated in opt.step().
    metamodel = l2l.algorithms.GBML(
        model, transform, allow_nograd=True, lr=0.001, adapt_transform=False
    )
    opt = torch.optim.Adam(metamodel.parameters(), lr=1.0)
    print("\nPre-learning")
    print("Transform parameters: ", transform)
    for name, p in metamodel.named_parameters():
        print(name, ":", p.norm())

    for task in range(10):
        opt.zero_grad()
        learner = metamodel.clone()
        X = torch.randn(N_SAMPLES, N_DIMS)

        # fast adapt
        for step in range(N_STEPS):
            adapt_loss = learner(X).norm(2)
            learner.adapt(adapt_loss)

        # meta-learn
        eval_loss = learner(X).norm(2)
        eval_loss.backward()
        opt.step()

    print("\nPost-learning")
    print("Transform parameters: ", transform)
    for name, p in metamodel.named_parameters():
        print(name, ":", p.norm())

    print("\n\n--------------------------")
    print("[*] Testing per-layer per-step LR with three linear layers")
    model = torch.nn.Sequential(
        torch.nn.Linear(N_DIMS, 24), torch.nn.Linear(24, 16), torch.nn.Linear(16, 10)
    )
    transform = PerLayerPerStepLRTransform(1e-3, N_STEPS, model)
    # Setting adapt_transform=True means that the transform will be updated in
    # the *adapt* function, which is not what we want. We want it to compute gradients during
    # eval_loss.backward() only, so that it's updated in opt.step().
    metamodel = l2l.algorithms.GBML(
        model,
        transform,
        allow_nograd=True,
        lr=0.001,
        adapt_transform=False,
        pass_param_names=True,
    )
    opt = torch.optim.Adam(metamodel.parameters(), lr=1.0)
    print("\nPre-learning")
    print("Transform parameters: ", transform)
    for name, p in metamodel.named_parameters():
        print(name, ":", p.norm())

    for task in range(10):
        opt.zero_grad()
        learner = metamodel.clone()
        X = torch.randn(N_SAMPLES, N_DIMS)

        # fast adapt
        for step in range(N_STEPS):
            adapt_loss = learner(X).norm(2)
            learner.adapt(adapt_loss)

        # meta-learn
        eval_loss = learner(X).norm(2)
        eval_loss.backward()
        opt.step()

    print("\nPost-learning")
    print("Transform parameters: ", transform)
    for name, p in metamodel.named_parameters():
        print(name, ":", p.norm())
