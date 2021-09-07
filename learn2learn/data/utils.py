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

    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/data/utils.py)

    **Description**

    Infinitely loops over a given iterator.

    **Arguments**

    * **dataloader** (iterator) - Iterator to loop over.

    **Example**
    ~~~python
    dataloader = DataLoader(dataset, shuffle=True, batch_size=32)
    inf_dataloader = InfiniteIterator(dataloader)
    for iteration in range(10000):  # guaranteed to reach 10,000 regardless of len(dataloader)
        X, y = next(inf_dataloader)
    ~~~
    """

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


def partition_task(data, labels, shots=1):

    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/data/utils.py)

    **Description**

    Partitions a classification task into support and query sets.

    The support set will contain `shots` samples per class, the query will take the remaining samples.

    Assumes each class in `labels` is associated with the same number of samples in `data`.

    **Arguments**

    * **data** (Tensor) - Data to be partitioned into support and query.
    * **labels** (Tensor) - Labels of each data sample, used for partitioning.
    * **shots** (int, *optional*, default=1) - Number of data samples per class in the support set.

    **Example**
    ~~~python
    X, y = taskset.sample()
    (X_support, y_support), (X_query, y_query) = partition_task(X, y, shots=5)
    ~~~
    """

    assert data.size(0) == labels.size(0)
    unique_labels = labels.unique()
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

    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/data/utils.py)

    **Description**

    Converts an entire dataset into a TensorDataset, and optionally puts it on the desired device.

    Useful to accelerate training with relatively small datasets.
    If the device is cpu and cuda is available, the TensorDataset will live in pinned memory.

    **Arguments**

    * **dataset** (Dataset) - Dataset to put on a device.
    * **device** (torch.device, *optional*, default=None) - Device of dataset. Defaults to CPU.
    * **transform** (transform, *optional*, default=None) - Transform to apply on the first variate of the dataset's samples X.

    **Example**
    ~~~python
    transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        lambda x: x.view(1, 28, 28),
    ])
    mnist = MNIST('~/data')
    mnist_ondevice = OnDeviceDataset(mnist, device='cuda', transform=transforms)
    mnist_meta = MetaDataset(mnist_ondevice)
    ~~~
    """

    def __init__(self, dataset, device=None, transform=None):
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
        if data.device == torch.device('cpu') and torch.cuda.is_available():
            data = data.pin_memory()
            labels = labels.pin_memory()
        super(OnDeviceDataset, self).__init__(data, labels)
        self.transform = transform
        if hasattr(dataset, '_bookkeeping_path'):
            self._bookkeeping_path = dataset._bookkeeping_path

    def __getitem__(self, index):
        x, y = super(OnDeviceDataset, self).__getitem__(index)
        if self.transform is not None:
            x = self.transform(x)
        return x, y
