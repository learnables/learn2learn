#!/usr/bin/env python3

import os
import io
import pickle
import tarfile

import numpy as np
import torch
import torch.utils.data as data

from PIL import Image

from learn2learn.data.utils import download_file_from_google_drive


class TieredImagenet(data.Dataset):
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/vision/datasets/tiered_imagenet.py)

    **Description**

    The *tiered*-ImageNet dataset was originally introduced by ...


    **References**

    1. 
    2. https://github.com/renmengye/few-shot-ssl-public 

    **Arguments**

    * **root** (str) - Path to download the data.
    * **mode** (str, *optional*, default='train') - Which split to use.
        Must be 'train', 'validation', or 'test'.
    * **transform** (Transform, *optional*, default=None) - Input pre-processing.
    * **target_transform** (Transform, *optional*, default=None) - Target pre-processing.

    **Example**

    ~~~python
    train_dataset = l2l.vision.datasets.TieredImagenet(root='./data', mode='train')
    train_dataset = l2l.data.MetaDataset(train_dataset)
    train_generator = l2l.data.TaskDataset(dataset=train_dataset, num_tasks=1000)
    ~~~

    """

    def __init__(self, root, mode='train', transform=None, target_transform=None, download=False):
        super(TieredImagenet, self).__init__()
        self.root = root
        if not os.path.exists(root):
            os.mkdir(root)
        self.transform = transform
        self.target_transform = target_transform
        if mode not in ['train', 'validation', 'test']:
            raise ValueError('mode must be train, validation, or test.')
        self.mode = mode
        self._bookkeeping_path = os.path.join(self.root, 'tiered-imagenet-bookkeeping-' + mode + '.pkl')
        google_drive_file_id = '1g1aIDy2Ar_MViF2gDXFYDBTR-HYecV07'

        if not self._check_exists() and download:
            self.download(google_drive_file_id, self.root)

        short_mode = 'val' if mode == 'validation' else mode
        tiered_imaganet_path = os.path.join(self.root, 'tiered-imagenet')
        images_path = os.path.join(tiered_imaganet_path, short_mode + '_images_png.pkl')
        with open(images_path, 'rb') as images_file:
            self.images = pickle.load(images_file)
        labels_path = os.path.join(tiered_imaganet_path, short_mode + '_labels.pkl')
        with open(labels_path, 'rb') as labels_file:
            self.labels = pickle.load(labels_file)
            self.labels = self.labels['label_specific']

    def download(self, file_id, destination):
        archive_path = os.path.join(destination, 'tiered_imagenet.tar')
        print('Downloading tiered ImageNet. (12Gb) Please be patient.')
        download_file_from_google_drive(file_id, archive_path)
        archive_file = tarfile.open(archive_path)
        archive_file.extractall(destination)
        os.remove(archive_path)

    def __getitem__(self, idx):
        image = Image.open(io.BytesIO(self.images[idx]))
        label = self.labels[idx]
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return image, label

    def __len__(self):
        return len(self.labels)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root,
                                           'tiered-imagenet',
                                           'train_images_png.pkl'))


if __name__ == '__main__':
    dataset = TieredImagenet(root=os.path.expanduser('~/data'))
    img, tgt = dataset[43]
    dataset = TieredImagenet(root=os.path.expanduser('~/data'), mode='validation')
    img, tgt = dataset[43]
    dataset = TieredImagenet(root=os.path.expanduser('~/data'), mode='test')
    img, tgt = dataset[43]
