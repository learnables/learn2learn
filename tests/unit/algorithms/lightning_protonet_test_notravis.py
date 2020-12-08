#!/usr/bin/env python3

import unittest
import learn2learn as l2l
import pytorch_lightning as pl

from learn2learn.utils.lightning import EpisodicBatcher
from learn2learn.algorithms import LightningPrototypicalNetworks


class TestLightningProtoNets(unittest.TestCase):

    def test_protonets(self):
        meta_batch_size = 4
        max_epochs = 20
        seed = 42
        ways = 5
        shots = 5

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

        # init model
        model = l2l.vision.models.CNN4(ways, embedding_size=32*4)
        features = model.features
        protonet = LightningPrototypicalNetworks(features, lr=3e-4)
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
        trainer.fit(protonet, episodic_data)
        acc = trainer.test(
            test_dataloaders=tasksets.test,
            verbose=False,
        )
        self.assertTrue(acc[0]["valid_accuracy"] >= 0.20)


if __name__ == "__main__":
    unittest.main()
