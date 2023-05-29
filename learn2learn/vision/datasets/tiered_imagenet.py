#!/usr/bin/env python3

import os
import io
import pickle
import tarfile

import numpy as np
import torch
import torch.utils.data as data

from PIL import Image

from learn2learn.data.utils import (
    download_file_from_google_drive,
    download_file,
)


class TieredImagenet(data.Dataset):
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/vision/datasets/tiered_imagenet.py)

    **Description**

    The *tiered*-ImageNet dataset was originally introduced by Ren et al, 2018 and we download the data directly from the link provided in their repository.

    Like *mini*-ImageNet, *tiered*-ImageNet builds on top of ILSVRC-12, but consists of 608 classes (779,165 images) instead of 100.
    The train-validation-test split is made such that classes from similar categories are in the same splits.
    There are 34 categories each containing between 10 and 30 classes.
    Of these categories, 20 (351 classes; 448,695 images) are used for training,
    6 (97 classes; 124,261 images) for validation, and 8 (160 class; 206,209 images) for testing.

    **References**

    1. Ren et al, 2018. "Meta-Learning for Semi-Supervised Few-Shot Classification." ICLR '18.
    2. Ren Mengye. 2018. "few-shot-ssl-public". [https://github.com/renmengye/few-shot-ssl-public](https://github.com/renmengye/few-shot-ssl-public)

    **Arguments**

    * **root** (str) - Path to download the data.
    * **mode** (str, *optional*, default='train') - Which split to use.
        Must be 'train', 'validation', or 'test'.
    * **transform** (Transform, *optional*, default=None) - Input pre-processing.
    * **target_transform** (Transform, *optional*, default=None) - Target pre-processing.
    * **download** (bool, *optional*, default=False) - Whether to download the dataset.

    **Example**

    ~~~python
    train_dataset = l2l.vision.datasets.TieredImagenet(root='./data', mode='train', download=True)
    train_dataset = l2l.data.MetaDataset(train_dataset)
    train_generator = l2l.data.Taskset(dataset=train_dataset, num_tasks=1000)
    ~~~

    """

    def __init__(self, root, mode='train', transform=None, target_transform=None, download=False):
        super(TieredImagenet, self).__init__()
        self.root = os.path.expanduser(root)
        if not os.path.exists(self.root):
            os.mkdir(self.root)
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
        print('Downloading tiered ImageNet. (12Gb) Please be patient.')
        try:
            archive_dir = os.path.join(destination, 'tiered-imagenet')
            os.makedirs(archive_dir, exist_ok=True)
            files_to_download = [
                'https://zenodo.org/record/7978538/files/tiered-imagenet-class_names.txt',
                'https://zenodo.org/record/7978538/files/tiered-imagenet-synsets.txt',
                'https://zenodo.org/record/7978538/files/tiered-imagenet-test_images_png.pkl',
                'https://zenodo.org/record/7978538/files/tiered-imagenet-test_labels.pkl',
                'https://zenodo.org/record/7978538/files/tiered-imagenet-train_images_png.pkl',
                'https://zenodo.org/record/7978538/files/tiered-imagenet-train_labels.pkl',
                'https://zenodo.org/record/7978538/files/tiered-imagenet-val_images_png.pkl',
                'https://zenodo.org/record/7978538/files/tiered-imagenet-val_labels.pkl',
            ]
            for file_url in files_to_download:
                file_dest = os.path.join(
                    archive_dir,
                    os.path.basename(file_url).replace('tiered-imagenet-', '')
                )
                download_file(
                    source=file_url,
                    destination=file_dest,
                )
        except Exception:
            archive_path = os.path.join(destination, 'tiered_imagenet.tar')
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
    dataset = TieredImagenet(root='~/data')
    img, tgt = dataset[43]
    dataset = TieredImagenet(root='~/data', mode='validation')
    img, tgt = dataset[43]
    dataset = TieredImagenet(root='~/data', mode='test')
    img, tgt = dataset[43]
