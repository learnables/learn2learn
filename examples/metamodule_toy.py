#!/usr/bin/env python

"""
A simple example of how `MetaModule` replaces submodules with new ones.

Here we simply append a layer norm after each linear layer and swap the relu activations for gelus.
"""

import torch
import learn2learn as l2l
import learn2learn.nn.metalayers as ml

if __name__ == "__main__":
    original = torch.nn.Sequential(
        torch.nn.Linear(10, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 10),
        torch.nn.ReLU(),
    )
    print('Original model:')
    print(original)
    print('\n')

    wrapped = ml.MetaModule(
        module=original,
        substitutions={
            torch.nn.Linear: lambda linear: torch.nn.Sequential(
                linear,
                torch.nn.LayerNorm(linear.in_features),
            ),
            torch.nn.ReLU: lambda relu: torch.nn.GELU(),
        },
        freeze_module=True,
    )
    print('Wrapped model:')
    print(wrapped)
    print('\n')

    # Note that the original parameters are frozen in wrapped:
    print('Parameters in wrapped which require gradients (only layer norms):')
    for name, param in wrapped.named_parameters():
        print(f'{name:<30} | {param.requires_grad!s:>6}')
