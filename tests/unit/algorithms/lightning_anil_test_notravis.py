#!/usr/bin/env python3

import unittest
import torch
import learn2learn as l2l
import pytorch_lightning as pl

from learn2learn.utils.lightning import EpisodicBatcher
from learn2learn.algorithms import LightningANIL


class Lambda(torch.nn.Module):
    def __init__(self, fn):
        super(Lambda, self).__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


class TestLightningANIL(unittest.TestCase):
    def test_anil(self):
        meta_batch_size = 32
        max_epochs = 5
        seed = 42
        ways = 5
        shots = 5
        adaptation_lr = 5e-1

        pl.seed_everything(seed)

        # Create tasksets using the benchmark interface
        tasksets = l2l.vision.benchmarks.get_tasksets(
            "cifarfs",
            train_samples=2 * shots,
            train_ways=ways,
            test_samples=2 * shots,
            test_ways=ways,
            root="~/data",
        )

        self.assertTrue(len(tasksets) == 3)

        features = l2l.vision.models.ConvBase(channels=3, max_pool=True)
        features = torch.nn.Sequential(features, Lambda(lambda x: x.view(-1, 256)))
        classifier = torch.nn.Linear(256, ways)
        # init model
        anil = LightningANIL(features, classifier, adaptation_lr=adaptation_lr)
        episodic_data = EpisodicBatcher(
            tasksets.train,
            tasksets.validation,
            tasksets.test,
            epoch_length=meta_batch_size,
        )

        trainer = pl.Trainer(
            accumulate_grad_batches=meta_batch_size,
            min_epochs=max_epochs,
            max_epochs=max_epochs,
            progress_bar_refresh_rate=0,
            deterministic=True,
            weights_summary=None,
        )
        trainer.fit(anil, episodic_data)
        acc = trainer.test(
            test_dataloaders=tasksets.test,
            verbose=False,
        )
        self.assertTrue(acc[0]["valid_accuracy"] >= 0.20)


if __name__ == "__main__":
    unittest.main()
