#!/usr/bin/env python3

import os
import numpy as np
import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset
import torchvision.transforms as transforms

from PIL import Image
from models import MAMLFC


"""
The proper dataset is the resized one, when dimensionality != 224.
(224 is standard imagenet sizes.)
"""


class MiniImagenet(TensorDataset):

    """
    Acts as a TensorDataset of mini-Imagenet images.
    """

    def __init__(self, path, transform, cuda=False):
        """Loads the images in memory"""
        print('Loading: ', path)
        self.path = path
        self.images = []
        self.labels = []

        # Go through each folder, load transformed images
        labels = os.listdir(path)
        for label, label_name in enumerate(labels):
            label_folder = os.path.join(path, label_name)
            for img_path in os.listdir(label_folder):
                img_path = os.path.join(label_folder, img_path)
                img = transform(img_path)
                self.labels.append(label)
                self.images.append(img)
        self.images = th.stack(self.images)
        self.labels = th.tensor(self.labels).view(-1, 1)
        if cuda:
            self.images = self.images.cuda()
            self.labels = self.labels.cuda()
        super(MiniImagenet, self).__init__(self.images, self.labels)



def get_miniimagenet(
        batch_size=1,
        lr=0.4,
        momentum=0.9,
        cuda=False,
        data_path='./data/',
        model=None,
        ways=5,
        use_mean=False,
        grayscale=False,
        dimensionality=84,
        preprocessed=False,
        weight_decay=0.0,
        ):

    data_folder = os.path.join(data_path, 'miniimagenet')
    if dimensionality == 224:
        data_folder = os.path.join(data_folder, 'resized224')
        scaling_fn = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])
    else:
        data_folder = os.path.join(data_folder, 'resized')
        scaling_fn = lambda x: x.float() / 255.
    train_images_path = os.path.join(data_folder, 'train')
    valid_images_path = os.path.join(data_folder, 'val')
    test_images_path = os.path.join(data_folder, 'test')

    transform = [lambda x: Image.open(x),
                 transforms.ToTensor(),
                 scaling_fn,
                ]
    transform = transforms.Compose(transform)

    train_set = MiniImagenet(train_images_path, transform, cuda)
    valid_set = MiniImagenet(valid_images_path, transform, cuda)
    test_set = MiniImagenet(test_images_path, transform, cuda)
