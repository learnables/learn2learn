
import requests
from __future__ import print_function
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


class MiniImagenetDataset(data.Dataset):
    def __init__(self, root, mode='train', transform=None, target_transform=None):
        '''
        The items are (filename,category). The index of all the categories can be found in self.idx_classes
        Args:
        - root: the directory where the dataset will be stored or downloaded to if not available
        - transform: how to transform the input
        - target_transform: how to transform the target
        '''
        super(MiniImagenetDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.mode = mode
        if self.mode == 'test':
            google_drive_file_id = '1wpmY-hmiJUUlRBkO9ZDCXAcIpHEFdOhD'
        elif self.mode == 'train':
            google_drive_file_id = '1I3itTXpXxGV68olxM5roceUMG8itH9Xj'
        elif self.mode == 'val':
            google_drive_file_id = '1KY5e491bkLFqJDp0-UWou3463Mo8AOco'
        else:
            raise ('ValueError', 'Needs to be train, test or val')

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

