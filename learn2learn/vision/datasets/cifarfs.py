#!/usr/bin/env python3

import os
import shutil
import zipfile
import numpy as np
import torch
import torch.utils.data as data

from torchvision.datasets import ImageFolder
from learn2learn.data.utils import (
    download_file_from_google_drive,
    download_file,
)


class CIFARFS(ImageFolder):

    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/vision/datasets/cifarfs.py)

    **Description**

    The CIFAR Few-Shot dataset as originally introduced by Bertinetto et al., 2019.

    It consists of 60'000 colour images of sizes 32x32 pixels.
    The dataset is divided in 3 splits of 64 training, 16 validation, and 20 testing classes each containing 600 examples.
    The classes are sampled from the CIFAR-100 dataset, and we use the splits from Bertinetto et al., 2019.

    **References**

    1. Bertinetto et al. 2019. "Meta-learning with differentiable closed-form solvers". ICLR.

    **Arguments**

    * **root** (str) - Path to download the data.
    * **mode** (str, *optional*, default='train') - Which split to use.
        Must be 'train', 'validation', or 'test'.
    * **transform** (Transform, *optional*, default=None) - Input pre-processing.
    * **target_transform** (Transform, *optional*, default=None) - Target pre-processing.

    **Example**

    ~~~python
    train_dataset = l2l.vision.datasets.CIFARFS(root='./data', mode='train')
    train_dataset = l2l.data.MetaDataset(train_dataset)
    train_generator = l2l.data.TaskGenerator(dataset=train_dataset, ways=ways)
    ~~~

    """

    def __init__(self,
                 root,
                 mode='train',
                 transform=None,
                 target_transform=None,
                 download=False):
        self.root = os.path.expanduser(root)
        if not os.path.exists(self.root):
            os.mkdir(self.root)
        self.transform = transform
        self.target_transform = target_transform
        self.mode = mode
        self.processed_root = os.path.join(self.root, 'cifarfs', 'processed')
        self.raw_path = os.path.join(self.root, 'cifarfs')

        if not self._check_exists() and download:
            self._download()
        if not self._check_processed():
            self._process_zip()
        mode = 'val' if mode == 'validation' else mode
        self.processed_root = os.path.join(self.processed_root, mode)
        self._bookkeeping_path = os.path.join(self.root, 'cifarfs-' + mode + '-bookkeeping.pkl')
        super(CIFARFS, self).__init__(root=self.processed_root,
                                      transform=self.transform,
                                      target_transform=self.target_transform)

    def _check_exists(self):
        return os.path.exists(self.raw_path)

    def _check_processed(self):
        return os.path.exists(self.processed_root)

    def _download(self):
        # Download the zip, unzip it, and clean up
        print('Downloading CIFARFS to ', self.root)
        if not os.path.exists(self.root):
            os.mkdir(self.root)
        zip_file = os.path.join(self.root, 'cifarfs.zip')
        try:
            download_file(
                source='https://zenodo.org/record/7978538/files/cifar100.zip',
                destination=zip_file,
            )
            with zipfile.ZipFile(zip_file, 'r') as zfile:
                zfile.extractall(self.raw_path)
            os.remove(zip_file)
        except Exception:
            download_file_from_google_drive('1pTsCCMDj45kzFYgrnO67BWVbKs48Q3NI',
                                            zip_file)
            with zipfile.ZipFile(zip_file, 'r') as zfile:
                zfile.extractall(self.raw_path)
            os.remove(zip_file)

    def _process_zip(self):
        print('Creating CIFARFS splits')
        if not os.path.exists(self.processed_root):
            os.mkdir(self.processed_root)
        split_path = os.path.join(self.raw_path, 'cifar100', 'splits', 'bertinetto')
        train_split_file = os.path.join(split_path, 'train.txt')
        valid_split_file = os.path.join(split_path, 'val.txt')
        test_split_file = os.path.join(split_path, 'test.txt')

        source_dir = os.path.join(self.raw_path, 'cifar100', 'data')
        for fname, dest in [(train_split_file, 'train'),
                            (valid_split_file, 'val'),
                            (test_split_file, 'test')]:
            dest_target = os.path.join(self.processed_root, dest)
            if not os.path.exists(dest_target):
                os.mkdir(dest_target)
            with open(fname) as split:
                for label in split.readlines():
                    source = os.path.join(source_dir, label.strip())
                    target = os.path.join(dest_target, label.strip())
                    shutil.copytree(source, target)


if __name__ == '__main__':
    cifarfs = CIFARFS('./data')
