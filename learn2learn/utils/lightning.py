#!/usr/bin/env python3

"""
Some utilities to interface with PyTorch Lightning.
"""

import pytorch_lightning as pl
import sys
import tqdm


class EpisodicBatcher(pl.LightningDataModule):

    """
    nc
    """

    def __init__(
        self,
        train_tasks,
        validation_tasks=None,
        test_tasks=None,
        epoch_length=1,
    ):
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
        length = self.epoch_length
        return EpisodicBatcher.epochify(
            self.test_tasks,
            length,
        )


class NoLeaveProgressBar(pl.callbacks.TQDMProgressBar):

    def init_test_tqdm(self):
        bar = tqdm.tqdm(
            desc='Testing',
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=False,
            dynamic_ncols=True,
            file=sys.stdout
        )
        return bar


class TrackTestAccuracyCallback(pl.callbacks.Callback):

    def on_validation_end(self, trainer, module):
        trainer.test(model=module, verbose=False)
