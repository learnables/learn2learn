#!/usr/bin/env python3

r"""
Standardized implementations of few-shot learning algorithms,
compatible with PyTorch Lightning.
"""

from .lightning_episodic_module import LightningEpisodicModule
from .lightning_maml import LightningMAML
from .lightning_anil import LightningANIL
from .lightning_protonet import LightningPrototypicalNetworks
from .lightning_metaoptnet import LightningMetaOptNet
