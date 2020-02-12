#!/usr/bin/env python3

import os
import tarfile
import requests
import scipy.io
from PIL import Image

from torch.utils.data import Dataset

DATA_DIR = 'vgg_flower102'
IMAGES_URL = 'http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz'
LABELS_URL = 'http://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat'
IMAGES_DIR = 'jpg'
LABELS_PATH = 'imagelabels.mat'

# Splits from "Meta-Datasets", Triantafillou et al, 2019
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

    def __init__(self, root, mode='all', transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self._bookkeeping_path = os.path.join(self.root, 'vgg-flower102-' + mode +'-bookkeeping.pkl')

        if not self._check_exists() and download:
            self.download(self.root)

        self.load_data(mode)

    def _check_exists(self):
        data_path = os.path.join(self.root, DATA_DIR)
        return os.path.exists(data_path)

    def download(self, root):
        if not os.path.exists(root):
            os.mkdir(root)
        data_path = os.path.join(root, DATA_DIR)
        if not os.path.exists(data_path):
            os.mkdir(data_path)
        tar_path = os.path.join(data_path, os.path.basename(IMAGES_URL))
        print('Downloading VGG Flower102 dataset')
        req = requests.get(IMAGES_URL)
        with open(tar_path, 'wb') as archive:
            archive.write(req.content)
        tar_file = tarfile.TarFile(tar_path)
        tar_file.extractall(data_path)
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
    flowers = VGGFlower102('~/data', download=True)
    print(len(flowers))
    import pdb; pdb.set_trace()
