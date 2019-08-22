#!/usr/bin/env python3

import numpy as np

import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms

from PIL import Image, ImageFilter

from omniglot import Omniglot
from models import MAMLFC
from utils import split_on_labels


def get_omniglot(
        batch_size=1,
        lr=0.4,
        momentum=0.9,
        cuda=False,
        data_path='./data/',
        model=None,
        ways=5,
        use_mean=True,
        weight_decay=0.0,
        ):
        transform = transforms.Compose([lambda x: Image.open(x).convert('L'),
                                        lambda x: x.resize((28, 28)),
                                        lambda x: np.reshape(x, (1, 28, 28)),
                                        lambda x: np.transpose(x, [1, 2, 0]),
                                        lambda x: np.float32(x) / 255.,
                                        lambda x: 1.0 - x,
                                        transforms.ToTensor()])
        target_transform = None
        omniglot = Omniglot(download=True, transform=transform,
                            target_transform=target_transform)
        labels = list(range(1623))
        num_test = len(labels) - 1200
        # NOTE the rotation here: the characters are randomly rotated 0/90/180/270 degrees.
        train_set, valid_set, test_set = split_on_labels(omniglot,
                                                         [1100, 100, num_test],
                                                         labels=labels,
                                                         rotations=[0, 1, 2, 3],
                                                         cuda=cuda)
        """
        split_on_labels does what TaskGenerator already does: randomly assign some classes
        to train/valid/test and split them into the respective datasets.
        """
