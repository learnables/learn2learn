#!/usr/bin/env python3

r"""
A set of utilities for data & tasks loading, preprocessing, and sampling.
"""

from . import transforms
from .meta_dataset import MetaDataset, UnionMetaDataset, FilteredMetaDataset
from .task_dataset import TaskDataset, DataDescription
from .utils import OnDeviceDataset, partition_task, InfiniteIterator
