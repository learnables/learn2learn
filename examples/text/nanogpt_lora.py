#!/usr/bin/env python

""" 
nanoGPT adapted with LoRA layers.
"""

import numpy as np
import torch
import transformers
import learn2learn as l2l

from nanogpt_model import GPT, GPTConfig


class LoraLinear(torch.nn.Module):

    """
    LoRA for Linear layers implementation.

    From: https://arxiv.org/pdf/2106.09685.pdf
    
    NOTE: proj_features is task-dependent and likely needs tuning.
    """

    def __init__(self, linear, proj_features=4, alpha=32, dropout=0.0):
        super(LoraLinear, self).__init__()
        self.linear = l2l.nn.freeze(linear)
        self.down_proj = torch.nn.Linear(linear.in_features, proj_features, bias=False)
        self.up_proj = torch.nn.Linear(proj_features, linear.out_features, bias=False)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.scaling = l2l.nn.Scale(alpha=alpha / proj_features)

    def forward(self, x):
        a = x
        a = self.dropout(a)
        a = self.down_proj(a)
        a = self.up_proj(a)
        a = self.scaling(a)
        return a + self.linear(x)


def main():

    config = GPTConfig()
    model = GPT(config)

    # Wrap with MetaModule to freeze and add LoRA to ALL linear layers.
    model = l2l.nn.MetaModule(
        model,
        substitutions={
            torch.nn.Linear: lambda linear: LoraLinear(
                linear,
                proj_features=4,
                alpha=32.0,
            ),
        },
        freeze_module=True,
    )

    # Unfreeze last classification layers
    l2l.nn.unfreeze(model.lm_head)


    # TODO: Proceed with finetuning as usual...


if __name__ == "__main__":
    main()
