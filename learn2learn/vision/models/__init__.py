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

from .cnn4 import (
    fc_init_,
    maml_init_,
    ConvBlock,
    ConvBase,
    OmniglotFC,
    OmniglotCNN,
    MiniImagenetCNN,
    CNN4,
)
from .resnet12 import ResNet12
