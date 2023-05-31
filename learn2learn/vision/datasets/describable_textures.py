#!/usr/bin/env python3

import os
import tarfile

from PIL import Image
from torch.utils.data import Dataset

from learn2learn.data.utils import download_file
from torchvision.datasets.folder import default_loader

DATA_DIR = 'describable_textures'
ARCHIVE_URL = 'https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz'

SPLITS = {
    'train': [
        'chequered',
        'braided',
        'interlaced',
        'matted',
        'honeycombed',
        'marbled',
        'veined',
        'frilly',
        'zigzagged',
        'cobwebbed',
        'pitted',
        'waffled',
        'fibrous',
        'flecked',
        'grooved',
        'potholed',
        'blotchy',
        'stained',
        'crystalline',
        'dotted',
        'striped',
        'swirly',
        'meshed',
        'bubbly',
        'studded',
        'pleated',
        'lacelike',
        'polka-dotted',
        'perforated',
        'freckled',
        'smeared',
        'cracked',
        'wrinkled',
    ],
    'test': [
        'banded',
        'bumpy',
        'crosshatched',
        'knitted',
        'sprinkled',
        'stratified',
        'woven',
    ],
    'validation': [
        'gauzy',
        'grid',
        'lined',
        'paisley',
        'porous',
        'scaly',
        'spiralled',
    ]
}


class DescribableTextures(Dataset):

    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/vision/datasets/describable_textures.py)

    **Description**

    The VGG Describable Textures dataset was originally introduced by Cimpoi et al., 2014
    and then re-purposed for few-shot learning in Triantafillou et al., 2020.

    The dataset consists of 5640 images organized according to 47 texture classes.
    Each class consists of 120 images between 300x300 and 640x640 pixels.
    Each image contains at least 90% of the texture.
    We follow the train-validation-test splits of Triantafillou et al., 2020.
    (33 classes for train, 7 for validation and test.)


    **References**

    1. Cimpoi et al. 2014. "Describing Textures in the Wild." CVPR'14.
    2. Triantafillou et al. 2020. "Meta-Dataset: A Dataset of Datasets for Learning to Learn from Few Examples." ICLR '20.
    3. [https://www.robots.ox.ac.uk/~vgg/data/dtd/](https://www.robots.ox.ac.uk/~vgg/data/dtd/)

    **Arguments**

    * **root** (str) - Path to download the data.
    * **mode** (str, *optional*, default='train') - Which split to use.
        Must be 'train', 'validation', or 'test'.
    * **transform** (Transform, *optional*, default=None) - Input pre-processing.
    * **target_transform** (Transform, *optional*, default=None) - Target pre-processing.
    * **download** (bool, *optional*, default=False) - Whether to download the dataset.

    **Example**

    ~~~python
    train_dataset = l2l.vision.datasets.DescribableTextures(root='./data', mode='train')
    train_dataset = l2l.data.MetaDataset(train_dataset)
    train_generator = l2l.data.Taskset(dataset=train_dataset, num_tasks=1000)
    ~~~

    """

    def __init__(
        self,
        root,
        mode='all',
        transform=None,
        target_transform=None,
        download=False,
            ):
        root = os.path.expanduser(root)
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self._bookkeeping_path = os.path.join(self.root, 'vgg-dtd-' + mode + '-bookkeeping.pkl')

        if not self._check_exists() and download:
            self.download()

        self.load_data(mode)

    def _check_exists(self):
        data_path = os.path.join(self.root, DATA_DIR, 'dtd')
        return os.path.exists(data_path)

    def download(self):
        if not os.path.exists(self.root):
            os.mkdir(self.root)
        data_path = os.path.join(self.root, DATA_DIR)
        if not os.path.exists(data_path):
            os.mkdir(data_path)
        tar_path = os.path.join(data_path, os.path.basename(ARCHIVE_URL))
        print('Downloading Describable Textures dataset (600Mb)')
        download_file(ARCHIVE_URL, tar_path)
        tar_file = tarfile.open(tar_path)
        tar_file.extractall(data_path)
        tar_file.close()
        os.remove(tar_path)

    def load_data(self, mode='train'):
        data_path = os.path.join(self.root, DATA_DIR, 'dtd', 'images')
        self.data = []
        splits = sum(SPLITS.values(), []) if mode == 'all' else SPLITS[mode]
        for class_idx, split in enumerate(splits):
            class_path = os.path.join(data_path, split)
            for img_path in os.listdir(class_path):
                if img_path == '.directory':
                    continue
                img_path = os.path.join(class_path, img_path)
                img = default_loader(img_path)
                self.data.append((img, class_idx))

    def __getitem__(self, i):
        image, label = self.data[i]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    dtd = DescribableTextures(root='~/data', download=True)
    __import__('pdb').set_trace()
