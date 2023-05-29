#!/usr/bin/env python
# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" 
OpenAI GPT model fine-tuning with Adapter layers.

Adapted from https://github.com/huggingface/transformers/blob/main/examples/legacy/run_openai_gpt.py
Itself adapted from https://github.com/huggingface/pytorch-openai-transformer-lm/blob/master/train.py
Itself adapted from https://github.com/openai/finetune-transformer-lm/blob/master/train.py
"""

import numpy as np
import torch
import transformers
import learn2learn as l2l

from transformers import (
    OpenAIGPTDoubleHeadsModel,
    OpenAIGPTTokenizer,
)


class AdapterLayer(torch.nn.Module):

    """
    AdapterLayer implementation.

    From: https://arxiv.org/pdf/1902.00751.pdf
    
    NOTE: proj_features is task-dependent and likely needs tuning.
    """

    def __init__(self, in_features, proj_features=128, nonlinearity=None):
        super(AdapterLayer, self).__init__()
        self.down_proj = torch.nn.Linear(in_features, proj_features)
        if nonlinearity is None:
            nonlinearity = torch.nn.GELU()
        self.nonlinearity = nonlinearity
        self.up_proj = torch.nn.Linear(proj_features, in_features)

    def forward(self, x):
        a = self.down_proj(x)
        a = self.nonlinearity(a)
        a = self.up_proj(a)
        return x + a


def main():
    """
    Entire script setup goes here. See:
    https://github.com/huggingface/transformers/blob/main/examples/legacy/run_openai_gpt.py
    """

    # BEGIN of changes to lines 174 to 183

    # Instantiate GPT-2 as in original script
    special_tokens = ["_start_", "_delimiter_", "_classify_"]
    tokenizer = OpenAIGPTTokenizer.from_pretrained('gpt2')
    tokenizer.add_tokens(special_tokens)
    special_tokens_ids = tokenizer.convert_tokens_to_ids(special_tokens)
    model = OpenAIGPTDoubleHeadsModel.from_pretrained('gpt2')
    model.resize_token_embeddings(len(tokenizer))

    # Wrap with MetaModule to freeze and add adapter layers.
    model = l2l.nn.MetaModule(
        model,
        substitutions={
            transformers.models.openai.modeling_openai.Attention: lambda attention: torch.nn.Sequential(
                attention,
                AdapterLayer(in_features=768, proj_features=128),
            ),
            transformers.models.openai.modeling_openai.MLP: lambda mlp: torch.nn.Sequential(
                mlp,
                AdapterLayer(in_features=768, proj_features=128),
            ),
        },
        freeze_module=True,
    )
    # Unfreeze last classification layers
    l2l.nn.unfreeze(model.wrapped_module.lm_head)
    l2l.nn.unfreeze(model.wrapped_module.multiple_choice_head)

    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # END of changes to lines 174 to 183

    """
    Training / evaluation goes here. See:
    https://github.com/huggingface/transformers/blob/main/examples/legacy/run_openai_gpt.py
    """


if __name__ == "__main__":
    main()
