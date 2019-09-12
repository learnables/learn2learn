#!/usr/bin/env python3

from __future__ import print_function
import requests
import torch.utils.data as data
import numpy as np
import os
import torch
import pickle


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def download_pkl(google_drive_id, data_root, mode):
    filename = 'mini-imagenet-cache-' + mode
    file_path = os.path.join(data_root, filename)

    if not os.path.exists(file_path + '.pkl'):
        print('Downloading:', file_path + '.pkl')
        download_file_from_google_drive(google_drive_id, file_path + '.pkl')
        print("Download finished")
    else:
        print("Data was already downloaded")


def index_classes(items):
    idx = {}
    for i in items:
        if (i not in idx):
            idx[i] = len(idx)
    return idx


class MiniImagenet(data.Dataset):

    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/vision/datasets/mini_imagenet.py)

    **Description**

    The *mini*-ImageNet dataset was originally introduced by Vinyals et al., 2016.


    It consists of 60'000 colour images of sizes 84x84 pixels.
    The dataset is divided in 3 splits of 64 training, 16 validation, and 20 testing classes each containing 600 examples.
    The classes are sampled from the ImageNet dataset, and we use the splits from Ravi & Larochelle, 2017.

    **References**

    1. Vinyals et al. 2016. “Matching Networks for One Shot Learning.” NeurIPS.
    2. Ravi and Larochelle. 2017. “Optimization as a Model for Few-Shot Learning.” ICLR.

    **Arguments**

    * **root** (str) - Path to download the data.
    * **mode** (str, *optional*, default='train') - Which split to use.
        Must be 'train', 'validation', or 'test'.
    * **transform** (Transform, *optional*, default=None) - Input pre-processing.
    * **target_transform** (Transform, *optional*, default=None) - Target pre-processing.

    **Example**

    ~~~python
    train_dataset = l2l.vision.datasets.MiniImagenet(root='./data', mode='train')
    train_dataset = l2l.data.MetaDataset(train_dataset)
    train_generator = l2l.data.TaskGenerator(dataset=train_dataset, ways=ways)
    ~~~

    """

    def __init__(self, root, mode='train', transform=None, target_transform=None):
        super(MiniImagenet, self).__init__()
        self.root = root
        if not os.path.exists(root):
            os.mkdir(root)
        self.transform = transform
        self.target_transform = target_transform
        self.mode = mode
        if self.mode == 'test':
            google_drive_file_id = '1wpmY-hmiJUUlRBkO9ZDCXAcIpHEFdOhD'
        elif self.mode == 'train':
            google_drive_file_id = '1I3itTXpXxGV68olxM5roceUMG8itH9Xj'
        elif self.mode == 'validation':
            google_drive_file_id = '1KY5e491bkLFqJDp0-UWou3463Mo8AOco'
        else:
            raise ('ValueError', 'Needs to be train, test or validation')

        if not self._check_exists():
            download_pkl(google_drive_file_id, root, mode)

        pickle_file = os.path.join(self.root, 'mini-imagenet-cache-' + mode + '.pkl')
        f = open(pickle_file, 'rb')
        self.data = pickle.load(f)

        self.x = torch.FloatTensor([np.transpose(x, (2, 0, 1)) for x in self.data['image_data']])
        self.y = [-1 for _ in range(len(self.x))]
        self.class_idx = index_classes(self.data['class_dict'].keys())
        for class_name, idxs in self.data['class_dict'].items():
            for idx in idxs:
                self.y[idx] = self.class_idx[class_name]

    def __getitem__(self, idx):
        x = self.x[idx]
        if self.transform:
            x = self.transform(x)
        return x, self.y[idx]

    def __len__(self):
        return len(self.x)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, 'mini-imagenet-cache-' + self.mode + '.pkl'))
