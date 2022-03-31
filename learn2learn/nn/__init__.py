#!/usr/bin/env python3

r"""
Additional `torch.nn.Module`s frequently used for meta-learning.
"""

from .kroneckers import *
from .misc import *
from .protonet import PrototypicalClassifier
from .metaoptnet import SVClassifier
from .metabatchnorm import MetaBatchNorm
