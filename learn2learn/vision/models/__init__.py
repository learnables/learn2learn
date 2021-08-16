#!/usr/bin/env python3

"""
**Description**

A set of commonly used models for meta-learning vision tasks.
For simplicity, all models' `forward` conform to the following API:

~~~python
def forward(self, x):
    x = self.features(x)
    x = self.classifier(x)
    return x
~~~
"""

import os
import torch

from learn2learn.data.utils import download_file

from .cnn4 import (
    fc_init_,
    maml_init_,
    ConvBlock,
    ConvBase,
    OmniglotFC,
    OmniglotCNN,
    MiniImagenetCNN,
    CNN4,
    CNN4Backbone,
)
from .resnet12 import ResNet12, ResNet12Backbone
from .wrn28 import WRN28, WRN28Backbone

__all__ = [
    'get_pretrained_backbone',
    'fc_init_',
    'maml_init_',
    'ConvBlock',
    'ConvBase',
    'OmniglotFC',
    'OmniglotCNN',
    'MiniImagenetCNN',
    'CNN4',
    'CNN4Backbone',
    'ResNet12',
    'ResNet12Backbone',
    'WRN28',
    'WRN28Backbone',
]

_BACKBONE_URLS = {
    'mini-imagenet': {
        'cnn4': 'https://zenodo.org/record/5204557/files/MiniImageNet-CNN4.pth',
        'resnet12': 'https://zenodo.org/record/5204557/files/MiniImageNet-ResNet12.pth',
        'wrn28': 'https://zenodo.org/record/5204557/files/MiniImageNet-WRN28.pth',
    },
    'tiered-imagenet': {
        'resnet12': 'https://zenodo.org/record/5204557/files/TieredImageNet-ResNet12.pth',
        'wrn28': 'https://zenodo.org/record/5204557/files/TieredImageNet-WRN28.pth',
    },
}


def get_pretrained_backbone(model, dataset, root, download=False):
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/vision/models/__init__.py)

    **Description**

    Returns pretrained backbone for a benchmark dataset.

    The returned object is a torch.nn.Module instance.

    **Arguments**

    * **model** (str) - The name of the model (`cnn4`, `resnet12`, or `wrn28`)
    * **dataset** (str) - The name of the benchmark dataset (`mini-imagenet` or `tiered-imagenet`).
    * **root** (str) - Location of the pretrained weights.
    * **download** (bool) - Download the pretrained weights if not available?

    **Example**
    ~~~python
    backbone = l2l.vision.models.get_pretrained_backbone(
        model='omniglot',
        dataset='mini-imagenet',
        root='~/.data',
        download=True,
    )
    ~~~
    """
    root = os.path.expanduser(root)
    destination_dir = os.path.join(root, 'pretrained_models', dataset)
    destination = os.path.join(destination_dir, model + '.pth')
    source = _BACKBONE_URLS[dataset][model]
    if not os.path.exists(destination) and download:
        print(f'Downloading {model} weights for {dataset}.')
        os.makedirs(destination_dir, exist_ok=True)
        download_file(source, destination)

    if model == 'cnn4':
        pretrained = CNN4Backbone(channels=3, max_pool=True)
    elif model == 'resnet12':
        pretrained = ResNet12Backbone(avg_pool=False)
    elif model == 'wrn28':
        pretrained = WRN28Backbone()

    weights = torch.load(destination)
    pretrained.load_state_dict(weights)
    return pretrained
