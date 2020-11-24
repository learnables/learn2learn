#!/usr/bin/env python3

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


# Define EpisodicBatcher for Lightning algorithms
try:
    import pytorch_lightning as pl

    class EpisodicBatcher(pl.LightningDataModule):
        def __init__(
            self,
            train_tasks,
            validation_tasks=None,
            test_tasks=None,
            epoch_length=1,
        ):
            """docstring for __init__"""
            super(EpisodicBatcher, self).__init__()
            self.train_tasks = train_tasks
            if validation_tasks is None:
                validation_tasks = train_tasks
            self.validation_tasks = validation_tasks
            if test_tasks is None:
                test_tasks = validation_tasks
            self.test_tasks = test_tasks
            self.epoch_length = epoch_length

        @staticmethod
        def epochify(taskset, epoch_length):
            class Epochifier(object):
                def __init__(self, tasks, length):
                    self.tasks = tasks
                    self.length = length

                def __getitem__(self, *args, **kwargs):
                    return self.tasks.sample()

                def __len__(self):
                    return self.length

            return Epochifier(taskset, epoch_length)

        def train_dataloader(self):
            return EpisodicBatcher.epochify(
                self.train_tasks,
                self.epoch_length,
            )

        def val_dataloader(self):
            return EpisodicBatcher.epochify(
                self.validation_tasks,
                self.epoch_length,
            )

        def test_dataloader(self):
            return EpisodicBatcher.epochify(
                self.test_tasks,
                self.epoch_length,
            )

except ImportError:
    from learn2learn.utils import _ImportRaiser
    EpisodicBatcher = _ImportRaiser(
        'pytorch_lightning',
        'pip install pytorch-lightning',
    )
