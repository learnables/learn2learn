#!/usr/bin/env python3

import os
import tarfile
import requests
import scipy.io

from PIL import Image
from torch.utils.data import Dataset

from learn2learn.data.utils import download_file

DATA_DIR = 'vgg_flower102'
IMAGES_URL = 'http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz'
LABELS_URL = 'http://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat'
IMAGES_DIR = 'jpg'
LABELS_PATH = 'imagelabels.mat'

# Splits from "Meta-Datasets", Triantafillou et al, 2020
SPLITS = {
    'train': [90, 38, 80, 30, 29, 12, 43, 27, 4, 64, 31, 99, 8, 67, 95, 77,
              78, 61, 88, 74, 55, 32, 21, 13, 79, 70, 51, 69, 14, 60, 11, 39,
              63, 37, 36, 28, 48, 7, 93, 2, 18, 24, 6, 3, 44, 76, 75, 72, 52,
              84, 73, 34, 54, 66, 59, 50, 91, 68, 100, 71, 81, 101, 92, 22,
              33, 87, 1, 49, 20, 25, 58],
    'validation': [10, 16, 17, 23, 26, 47, 53, 56, 57, 62, 82, 83, 86, 97, 102],
    'test': [5, 9, 15, 19, 35, 40, 41, 42, 45, 46, 65, 85, 89, 94, 96, 98],
    'all': list(range(1, 103)),
}


class VGGFlower102(Dataset):

    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/vision/datasets/vgg_flowers.py)

    **Description**

    The VGG Flowers dataset was originally introduced by Nilsback and Zisserman, 2006 and then re-purposed for few-shot learning in Triantafillou et al., 2020.

    The dataset consists of 102 classes of flowers, with each class consisting of 40 to 258 images.
    We provide the raw (unprocessed) images, and follow the train-validation-test splits of Triantafillou et al.

    **References**

    1. Nilsback, M. and A. Zisserman. 2006. "A Visual Vocabulary for Flower Classification." CVPR '06.
    2. Triantafillou et al. 2020. "Meta-Dataset: A Dataset of Datasets for Learning to Learn from Few Examples." ICLR '20.
    3. [https://www.robots.ox.ac.uk/~vgg/data/flowers/](https://www.robots.ox.ac.uk/~vgg/data/flowers/)

    **Arguments**

    * **root** (str) - Path to download the data.
    * **mode** (str, *optional*, default='train') - Which split to use.
        Must be 'train', 'validation', or 'test'.
    * **transform** (Transform, *optional*, default=None) - Input pre-processing.
    * **target_transform** (Transform, *optional*, default=None) - Target pre-processing.
    * **download** (bool, *optional*, default=False) - Whether to download the dataset.

    **Example**

    ~~~python
    train_dataset = l2l.vision.datasets.VGGFlower102(root='./data', mode='train')
    train_dataset = l2l.data.MetaDataset(train_dataset)
    train_generator = l2l.data.Taskset(dataset=train_dataset, num_tasks=1000)
    ~~~

    """

    def __init__(self, root, mode='all', transform=None, target_transform=None, download=False):
        root = os.path.expanduser(root)
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self._bookkeeping_path = os.path.join(self.root, 'vgg-flower102-' + mode + '-bookkeeping.pkl')

        if not self._check_exists() and download:
            self.download()

        self.load_data(mode)

    def _check_exists(self):
        data_path = os.path.join(self.root, DATA_DIR)
        return os.path.exists(data_path)

    def download(self):
        if not os.path.exists(self.root):
            os.mkdir(self.root)
        data_path = os.path.join(self.root, DATA_DIR)
        if not os.path.exists(data_path):
            os.mkdir(data_path)
        tar_path = os.path.join(data_path, os.path.basename(IMAGES_URL))
        print('Downloading VGG Flower102 dataset (330Mb)')
        download_file(IMAGES_URL, tar_path)
        tar_file = tarfile.open(tar_path)
        tar_file.extractall(data_path)
        tar_file.close()
        os.remove(tar_path)

        label_path = os.path.join(data_path, os.path.basename(LABELS_URL))
        req = requests.get(LABELS_URL)
        with open(label_path, 'wb') as label_file:
            label_file.write(req.content)

    def load_data(self, mode='train'):
        data_path = os.path.join(self.root, DATA_DIR)
        images_path = os.path.join(data_path, IMAGES_DIR)
        labels_path = os.path.join(data_path, LABELS_PATH)
        labels_mat = scipy.io.loadmat(labels_path)
        image_labels = []
        split = SPLITS[mode]
        for idx, label in enumerate(labels_mat['labels'][0], start=1):
            if label in split:
                image = str(idx).zfill(5)
                image = f'image_{image}.jpg'
                image = os.path.join(images_path, image)
                label = split.index(label)
                image_labels.append((image, label))
        self.data = image_labels

    def __getitem__(self, i):
        image, label = self.data[i]
        image = Image.open(image)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    assert len(SPLITS['train']) == 71
    assert len(SPLITS['validation']) == 15
    assert len(SPLITS['test']) == 16
    assert len(SPLITS['all']) == 102
    flowers = VGGFlower102('~/vgg_data', download=True)
    print(len(flowers))
