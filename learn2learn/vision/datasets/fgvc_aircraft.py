#!/usr/bin/env python3

import os
import pickle
import tarfile
import requests
from PIL import Image

from torch.utils.data import Dataset

DATASET_DIR = 'fgvc_aircraft'
DATASET_URL = 'http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz'
DATA_DIR = os.path.join('fgvc-aircraft-2013b', 'data')
IMAGES_DIR = os.path.join(DATA_DIR, 'images')
LABELS_PATH = os.path.join(DATA_DIR, 'labels.pkl')

# Splits from "Meta-Datasets", Triantafillou et al, 2020
SPLITS = {
    'train': ['A340-300', 'A318', 'Falcon 2000', 'F-16A/B', 'F/A-18', 'C-130',
              'MD-80', 'BAE 146-200', '777-200', '747-400', 'Cessna 172',
              'An-12', 'A330-300', 'A321', 'Fokker 100', 'Fokker 50', 'DHC-1',
              'Fokker 70', 'A340-200', 'DC-6', '747-200', 'Il-76', '747-300',
              'Model B200', 'Saab 340', 'Cessna 560', 'Dornier 328', 'E-195',
              'ERJ 135', '747-100', '737-600', 'C-47', 'DR-400', 'ATR-72',
              'A330-200', '727-200', '737-700', 'PA-28', 'ERJ 145', '737-300',
              '767-300', '737-500', '737-200', 'DHC-6', 'Falcon 900', 'DC-3',
              'Eurofighter Typhoon', 'Challenger 600', 'Hawk T1', 'A380',
              '777-300', 'E-190', 'DHC-8-100', 'Cessna 525', 'Metroliner',
              'EMB-120', 'Tu-134', 'Embraer Legacy 600', 'Gulfstream IV',
              'Tu-154', 'MD-87', 'A300B4', 'A340-600', 'A340-500', 'MD-11',
              '707-320', 'Cessna 208', 'Global Express', 'A319', 'DH-82'
              ],
    'test': ['737-400', '737-800', '757-200', '767-400', 'ATR-42', 'BAE-125',
             'Beechcraft 1900', 'Boeing 717', 'CRJ-200', 'CRJ-700', 'E-170',
             'L-1011', 'MD-90', 'Saab 2000', 'Spitfire'
             ],
    'valid': ['737-900', '757-300', '767-200', 'A310', 'A320', 'BAE 146-300',
              'CRJ-900', 'DC-10', 'DC-8', 'DC-9-30', 'DHC-8-300', 'Gulfstream V',
              'SR-20', 'Tornado', 'Yak-42'
              ],
    'all': ['A340-300', 'A318', 'Falcon 2000', 'F-16A/B', 'F/A-18', 'C-130',
            'MD-80', 'BAE 146-200', '777-200', '747-400', 'Cessna 172',
            'An-12', 'A330-300', 'A321', 'Fokker 100', 'Fokker 50', 'DHC-1',
            'Fokker 70', 'A340-200', 'DC-6', '747-200', 'Il-76', '747-300',
            'Model B200', 'Saab 340', 'Cessna 560', 'Dornier 328', 'E-195',
            'ERJ 135', '747-100', '737-600', 'C-47', 'DR-400', 'ATR-72',
            'A330-200', '727-200', '737-700', 'PA-28', 'ERJ 145', '737-300',
            '767-300', '737-500', '737-200', 'DHC-6', 'Falcon 900', 'DC-3',
            'Eurofighter Typhoon', 'Challenger 600', 'Hawk T1', 'A380',
            '777-300', 'E-190', 'DHC-8-100', 'Cessna 525', 'Metroliner',
            'EMB-120', 'Tu-134', 'Embraer Legacy 600', 'Gulfstream IV',
            'Tu-154', 'MD-87', 'A300B4', 'A340-600', 'A340-500', 'MD-11',
            '707-320', 'Cessna 208', 'Global Express', 'A319', 'DH-82',
            '737-900', '757-300', '767-200', 'A310', 'A320', 'BAE 146-300',
            'CRJ-900', 'DC-10', 'DC-8', 'DC-9-30', 'DHC-8-300', 'Gulfstream V',
            'SR-20', 'Tornado', 'Yak-42',
            '737-400', '737-800', '757-200', '767-400', 'ATR-42', 'BAE-125',
            'Beechcraft 1900', 'Boeing 717', 'CRJ-200', 'CRJ-700', 'E-170',
            'L-1011', 'MD-90', 'Saab 2000', 'Spitfire',
            ],
}


class FGVCAircraft(Dataset):

    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/vision/datasets/fgvc_aircraft.py)

    **Description**

    The FGVC Aircraft dataset was originally introduced by Maji et al., 2013 and then re-purposed for few-shot learning in Triantafillou et al., 2020.

    The dataset consists of 10,200 images of aircraft (102 classes, each 100 images).
    We provided the raw (un-processed) images and follow the train-validation-test splits of Triantafillou et al.
    TODO: Triantafillou et al. recommend cropping the images using the bounding box information,
    to remove copyright information and ensure that only one plane is visible in the image.

    **References**

    1. Maji et al. 2013. "Fine-Grained Visual Classification of Aircraft." arXiv [cs.CV].
    2. Triantafillou et al. 2020. "Meta-Dataset: A Dataset of Datasets for Learning to Learn from Few Examples." ICLR '20.
    3. [http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/](http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/)

    **Arguments**

    * **root** (str) - Path to download the data.
    * **mode** (str, *optional*, default='train') - Which split to use.
        Must be 'train', 'validation', or 'test'.
    * **transform** (Transform, *optional*, default=None) - Input pre-processing.
    * **target_transform** (Transform, *optional*, default=None) - Target pre-processing.
    * **download** (bool, *optional*, default=False) - Whether to download the dataset.
    * **bounding_box_crop** (bool, *optional*, default=False) - Whether to crop each image using bounding box information.

    **Example**

    ~~~python
    train_dataset = l2l.vision.datasets.FGVCAircraft(root='./data', mode='train', download=True)
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
        bounding_box_crop=False,
    ):
        root = os.path.expanduser(root)
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.bounding_box_crop = bounding_box_crop
        self._bookkeeping_path = os.path.join(
            self.root, 'fgvc-aircraft-' + mode + '-bookkeeping.pkl')

        if not self._check_exists() and download:
            self.download()

        assert mode in ['train', 'validation', 'test', 'all'], \
            'mode should be one of: train, validation, test, all.'
        self.load_data(mode)

    def _check_exists(self):
        data_path = os.path.join(self.root, DATASET_DIR)
        images_path = os.path.join(data_path, IMAGES_DIR)
        labels_path = os.path.join(data_path, LABELS_PATH)
        return os.path.exists(data_path) and \
            os.path.exists(images_path) and \
            os.path.exists(labels_path)

    def download(self):
        if not os.path.exists(self.root):
            os.mkdir(self.root)
        data_path = os.path.join(self.root, DATASET_DIR)
        if not os.path.exists(data_path):
            os.mkdir(data_path)
        tar_path = os.path.join(data_path, os.path.basename(DATASET_URL))
        if not os.path.exists(tar_path):
            print('Downloading FGVC Aircraft dataset. (2.75Gb)')
            req = requests.get(DATASET_URL)
            with open(tar_path, 'wb') as archive:
                for chunk in req.iter_content(chunk_size=32768):
                    if chunk:
                        archive.write(chunk)
        with tarfile.open(tar_path) as tar_file:
            def is_within_directory(directory, target):

                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)

                prefix = os.path.commonprefix([abs_directory, abs_target])

                return prefix == abs_directory

            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):

                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")

                tar.extractall(path, members, numeric_owner=numeric_owner)

            safe_extract(tar_file, data_path)
        family_names = [
            'images_variant_train.txt',
            'images_variant_val.txt',
            'images_variant_test.txt',
        ]
        images_labels = []
        for family in family_names:
            with open(os.path.join(data_path, DATA_DIR, family), 'r') as family_file:
                for line in family_file.readlines():
                    image, label = line.split(' ', 1)
                    images_labels.append((image.strip(), label.strip()))
        labels_path = os.path.join(data_path, LABELS_PATH)
        with open(labels_path, 'wb') as labels_file:
            pickle.dump(images_labels, labels_file)
        os.remove(tar_path)

    def load_data(self, mode='train'):
        data_path = os.path.join(self.root, DATASET_DIR)
        labels_path = os.path.join(data_path, LABELS_PATH)
        with open(labels_path, 'rb') as labels_file:
            image_labels = pickle.load(labels_file)

        data = []
        mode = 'valid' if mode == 'validation' else mode
        split = SPLITS[mode]

        # parse bounding boxes
        if self.bounding_box_crop:
            self.bounding_boxes = {}
            bbox_file = os.path.join(data_path, DATA_DIR, 'images_box.txt')
            bbox_content = {}
            with open(bbox_file, 'r') as bbox_fd:
                content = bbox_fd.readlines()
                for line in content:
                    line = line.split(' ')
                    bbox_content[line[0]] = (
                        int(line[1]),
                        int(line[2]),
                        int(line[3]),
                        int(line[4]),
                    )

        # read images from disk
        for image, label in image_labels:
            if label in split:
                image_path = os.path.join(
                    data_path, IMAGES_DIR, image + '.jpg')
                if self.bounding_box_crop:
                    self.bounding_boxes[image_path] = bbox_content[image]
                label = split.index(label)
                data.append((image_path, label))
        self.data = data

    def __getitem__(self, i):
        image_path, label = self.data[i]
        image = Image.open(image_path).convert('RGB')
        if self.bounding_box_crop:
            bbox = self.bounding_boxes[image_path]
            image = image.crop(bbox)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    assert len(SPLITS['all']) == len(SPLITS['train']) + \
        len(SPLITS['valid']) + len(SPLITS['test'])
    aircraft = FGVCAircraft('~/data', download=True, bounding_box_crop=True)
    img = aircraft[0]
    print(len(aircraft))

    import numpy as np
    import tqdm
    print('cropped:')
    data = FGVCAircraft('~/data', download=True, bounding_box_crop=True)
    min_size = float('inf')
    for img, label in tqdm.tqdm(data):
        min_size = min(min_size, *np.array(img).shape[:2])
    print('min_size:', min_size)

    data = FGVCAircraft(
        root="test_data/",
        mode="all",
        download=True,
        bounding_box_crop=False,
    )

    label_set = set()

    with tqdm.tqdm(total=len(data)) as pbar:
        for item in data:
            label_set.add(item[1])
            pbar.update(1)
            pbar.set_description(f"Found {len(label_set)} labels")
