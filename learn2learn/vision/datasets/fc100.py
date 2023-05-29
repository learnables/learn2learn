#!/usr/bin/env python3

import os
import pickle
import zipfile

import torch.utils.data as data

from PIL import Image

from learn2learn.data.utils import (
    download_file_from_google_drive,
    download_file,
)


class FC100(data.Dataset):
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/vision/datasets/fc100.py)

    **Description**

    The FC100 dataset was originally introduced by Oreshkin et al., 2018.

    It is based on CIFAR100, but unlike CIFAR-FS training, validation, and testing classes are
    split so as to minimize the information overlap between splits.
    The 100 classes are grouped into 20 superclasses of which 12 (60 classes) are used for training,
    4 (20 classes) for validation, and 4 (20 classes) for testing.
    Each class contains 600 images.
    The specific splits are provided in the Supplementary Material of the paper.
    Our data is downloaded from the link provided by [2].

    **References**

    1. Oreshkin et al. 2018. "TADAM: Task Dependent Adaptive Metric for Improved Few-Shot Learning." NeurIPS.
    2. Kwoonjoon Lee. 2019. "MetaOptNet." [https://github.com/kjunelee/MetaOptNet](https://github.com/kjunelee/MetaOptNet)

    **Arguments**

    * **root** (str) - Path to download the data.
    * **mode** (str, *optional*, default='train') - Which split to use.
        Must be 'train', 'validation', or 'test'.
    * **transform** (Transform, *optional*, default=None) - Input pre-processing.
    * **target_transform** (Transform, *optional*, default=None) - Target pre-processing.

    **Example**

    ~~~python
    train_dataset = l2l.vision.datasets.FC100(root='./data', mode='train')
    train_dataset = l2l.data.MetaDataset(train_dataset)
    train_generator = l2l.data.Taskset(dataset=train_dataset, num_tasks=1000)
    ~~~

    """

    GOOGLE_DRIVE_FILE_ID = '1_ZsLyqI487NRDQhwvI7rg86FK3YAZvz1'
    DROPBOX_LINK = 'https://www.dropbox.com/s/ftsjuwsu6lfp0fz/FC100.zip?dl=1'
    ZENODO_LINK = 'https://zenodo.org/record/7978538/files/fc100.zip'

    def __init__(self,
                 root,
                 mode='train',
                 transform=None,
                 target_transform=None,
                 download=False):
        super(FC100, self).__init__()
        self.root = os.path.expanduser(root)
        os.makedirs(self.root, exist_ok=True)
        self.transform = transform
        self.target_transform = target_transform
        if mode not in ['train', 'validation', 'test']:
            raise ValueError('mode must be train, validation, or test.')
        self.mode = mode
        self._bookkeeping_path = os.path.join(self.root, 'fc100-bookkeeping-' + mode + '.pkl')

        if not self._check_exists() and download:
            self.download()

        short_mode = 'val' if mode == 'validation' else mode
        fc100_path = os.path.join(self.root, 'FC100_' + short_mode + '.pickle')
        with open(fc100_path, 'rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            archive = u.load()
        self.images = archive['data']
        self.labels = archive['labels']

    def download(self):
        archive_path = os.path.join(self.root, 'fc100.zip')
        print('Downloading FC100. (160Mb)')
        try:
            download_file(FC100.ZENODO_LINK, archive_path)
            archive_file = zipfile.ZipFile(archive_path)
            archive_file.extractall(self.root)
            os.remove(archive_path)
        except Exception:
            try:  # Download from Google Drive first
                download_file_from_google_drive(FC100.GOOGLE_DRIVE_FILE_ID,
                                                archive_path)
                archive_file = zipfile.ZipFile(archive_path)
                archive_file.extractall(self.root)
                os.remove(archive_path)
            except zipfile.BadZipFile:
                download_file(FC100.DROPBOX_LINK, archive_path)
                archive_file = zipfile.ZipFile(archive_path)
                archive_file.extractall(self.root)
                os.remove(archive_path)

    def __getitem__(self, idx):
        image = self.images[idx]
        image = Image.fromarray(image)
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
                                           'FC100_train.pickle'))


if __name__ == '__main__':
    __import__('pdb').set_trace()
    dataset = FC100(root='~/data')
    img, tgt = dataset[43]
    dataset = FC100(root='~/data', mode='validation')
    img, tgt = dataset[43]
    dataset = FC100(root='~/data', mode='test')
    img, tgt = dataset[43]
