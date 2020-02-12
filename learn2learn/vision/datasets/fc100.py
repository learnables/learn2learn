#!/usr/bin/env python3

import os
import pickle
import zipfile

import torch.utils.data as data

from PIL import Image

from learn2learn.data.utils import download_file_from_google_drive


class FC100(data.Dataset):
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/vision/datasets/mini_imagenet.py)

    **Description**

    The FC100 dataset was originally introduced by...


    **References**

    1.
    2. https://github.com/kjunelee/MetaOptNet

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
    train_generator = l2l.data.TaskDataset(dataset=train_dataset, num_tasks=1000)
    ~~~

    """

    def __init__(self, root, mode='train', transform=None, target_transform=None):
        super(FC100, self).__init__()
        self.root = root
        if not os.path.exists(root):
            os.mkdir(root)
        self.transform = transform
        self.target_transform = target_transform
        if mode not in ['train', 'validation', 'test']:
            raise ValueError('mode must be train, validation, or test.')
        self.mode = mode
        self._bookkeeping_path = os.path.join(self.root, 'fc100-bookkeeping-' + mode + '.pkl')
        google_drive_file_id = '1_ZsLyqI487NRDQhwvI7rg86FK3YAZvz1'

        if not self._check_exists():
            self.download(google_drive_file_id, self.root)

        short_mode = 'val' if mode == 'validation' else mode
        fc100_path = os.path.join(self.root, 'FC100_' + short_mode + '.pickle')
        with open(fc100_path, 'rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            archive = u.load()
        self.images = archive['data']
        self.labels = archive['labels']

    def download(self, file_id, destination):
        archive_path = os.path.join(destination, 'fc100.zip')
        print('Downloading FC100. (160Mb) Please be patient.')
        download_file_from_google_drive(file_id, archive_path)
        archive_file = zipfile.ZipFile(archive_path)
        archive_file.extractall(destination)
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
    dataset = FC100(root=os.path.expanduser('~/data'))
    img, tgt = dataset[43]
    dataset = FC100(root=os.path.expanduser('~/data'), mode='validation')
    img, tgt = dataset[43]
    dataset = FC100(root=os.path.expanduser('~/data'), mode='test')
    img, tgt = dataset[43]
