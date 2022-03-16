#!/usr/bin/env python3

"""
"""

try:
    from pytorch_lightning import LightningModule
except ImportError:
    from learn2learn.utils import _ImportRaiser

    class LightningRaiser(_ImportRaiser):

        def __init__(self, **kwargs):
            name = 'pytorch_lightning'
            command = 'pip install pytorch_lightning'
            super(LightningRaiser, self).__init__(name, command)

    LightningModule = LightningRaiser

from torch import optim
from argparse import ArgumentParser


class LightningEpisodicModule(LightningModule):
    """docstring for LightningEpisodicModule"""

    train_shots = 1
    train_queries = 1
    train_ways = 5
    test_shots = 1
    test_queries = 1
    test_ways = 5
    lr = 0.001
    scheduler_step = 20
    scheduler_decay = 1.0

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(
            parents=[parent_parser],
            add_help=True,
            conflict_handler="resolve",
        )
        parser.add_argument(
            "--train_ways", type=int, default=LightningEpisodicModule.train_ways
        )
        parser.add_argument(
            "--train_shots", type=int, default=LightningEpisodicModule.train_shots
        )
        parser.add_argument(
            "--train_queries", type=int, default=LightningEpisodicModule.train_queries
        )
        parser.add_argument(
            "--test_ways", type=int, default=LightningEpisodicModule.test_ways
        )
        parser.add_argument(
            "--test_shots", type=int, default=LightningEpisodicModule.test_shots
        )
        parser.add_argument(
            "--test_queries", type=int, default=LightningEpisodicModule.test_queries
        )
        parser.add_argument("--lr", type=float, default=LightningEpisodicModule.lr)
        parser.add_argument(
            "--scheduler_step", type=int, default=LightningEpisodicModule.scheduler_step
        )
        parser.add_argument(
            "--scheduler_decay",
            type=float,
            default=LightningEpisodicModule.scheduler_decay,
        )
        return parser

    def training_step(self, batch, batch_idx):
        train_loss, train_accuracy = self.meta_learn(
            batch, batch_idx, self.train_ways, self.train_shots, self.train_queries
        )
        self.log(
            "train_loss",
            train_loss.item(),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.log(
            "train_accuracy",
            train_accuracy.item(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return train_loss

    def validation_step(self, batch, batch_idx):
        valid_loss, valid_accuracy = self.meta_learn(
            batch, batch_idx, self.test_ways, self.test_shots, self.test_queries
        )
        self.log(
            "valid_loss",
            valid_loss.item(),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.log(
            "valid_accuracy",
            valid_accuracy.item(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return valid_loss.item()

    def test_step(self, batch, batch_idx):
        test_loss, test_accuracy = self.meta_learn(
            batch, batch_idx, self.test_ways, self.test_shots, self.test_queries
        )
        self.log(
            "test_loss",
            test_loss.item(),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.log(
            "test_accuracy",
            test_accuracy.item(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return test_loss.item()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.scheduler_step,
            gamma=self.scheduler_decay,
        )
        return [optimizer], [lr_scheduler]
