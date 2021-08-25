#!/usr/bin/env python3

import torch
import requests
import tqdm

CHUNK_SIZE = 1 * 1024 * 1024


def download_file(source, destination, size=None):
    if size is None:
        size = 0
    req = requests.get(source, stream=True)
    with open(destination, 'wb') as archive:
        for chunk in tqdm.tqdm(
            req.iter_content(chunk_size=CHUNK_SIZE),
            total=size // CHUNK_SIZE,
            leave=False,
        ):
            if chunk:
                archive.write(chunk)


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


class InfiniteIterator(object):

    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = iter(self.dataloader)

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            try:
                return next(self.iterator)
            except StopIteration:
                self.iterator = iter(self.dataloader)


def partition_task(data, labels, shots=1, ways=None):
    assert data.size(0) == labels.size(0)
    unique_labels = labels.unique()
    if ways is None:
        ways = unique_labels.numel()
    data_shape = data.shape[1:]
    num_support = ways * shots
    num_query = data.size(0) - num_support
    assert num_query % ways == 0, 'Only query_shot == support_shot supported.'
    query_shots = num_query // ways
    support_data = torch.empty(
        (num_support,) + data_shape,
        device=data.device,
        dtype=data.dtype,
    )
    support_labels = torch.empty(
        num_support,
        device=labels.device,
        dtype=labels.dtype,
    )
    query_data = torch.empty(
        (num_query, ) + data_shape,
        device=data.device,
        dtype=data.dtype,
    )
    query_labels = torch.empty(
        num_query,
        device=labels.device,
        dtype=labels.dtype,
    )
    for i, label in enumerate(unique_labels):
        support_start = i * shots
        support_end = support_start + shots
        query_start = i * query_shots
        query_end = query_start + query_shots

        # filter data
        label_data = data[labels == label]  # TODO: fancy index makes a copy.
        num_label_data = label_data.size(0)
        assert num_label_data == shots + query_shots, \
            'Only same number of query per label supported.'

        # set value of labels
        support_labels[support_start:support_end].fill_(label)
        query_labels[query_start:query_end].fill_(label)

        # set value of data
        support_data[support_start:support_end].copy_(label_data[:shots])
        query_data[query_start:query_end].copy_(label_data[shots:])

    return (support_data, support_labels), (query_data, query_labels)


class OnDeviceDataset(torch.utils.data.TensorDataset):

    def __init__(self, dataset, device=None):
        data = []
        labels = []
        for x, y in dataset:
            data.append(x.unsqueeze(0))
            labels.append(y)
        data = torch.cat(data, dim=0)
        labels = torch.tensor(labels)
        if device is not None:
            data = data.to(device)
            labels = labels.to(device)
        super(OnDeviceDataset, self).__init__(data, labels)
