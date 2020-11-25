#!/usr/bin/env python3

"""
"""

import numpy as np
import torch
import learn2learn as l2l

from learn2learn.utils import accuracy
from learn2learn.algorithms.lightning import (
    LightningEpisodicModule,
    LightningMAML,
)


class LightningANIL(LightningEpisodicModule):

    """
    """

    def __init__(self, features, classifier, loss=None, **kwargs):
        super(LightningANIL, self).__init__()
        if loss is None:
            loss = torch.nn.CrossEntropyLoss(reduction="mean")
        self.loss = loss
        self.train_ways = kwargs.get("train_ways", LightningEpisodicModule.train_ways)
        self.train_shots = kwargs.get(
            "train_shots", LightningEpisodicModule.train_shots
        )
        self.train_queries = kwargs.get(
            "train_queries", LightningEpisodicModule.train_queries
        )
        self.test_ways = kwargs.get("test_ways", LightningEpisodicModule.test_ways)
        self.test_shots = kwargs.get("test_shots", LightningEpisodicModule.test_shots)
        self.test_queries = kwargs.get(
            "test_queries", LightningEpisodicModule.test_queries
        )
        self.lr = kwargs.get("lr", LightningEpisodicModule.lr)
        self.scheduler_step = kwargs.get(
            "scheduler_step", LightningEpisodicModule.scheduler_step
        )
        self.scheduler_decay = kwargs.get(
            "scheduler_decay", LightningEpisodicModule.scheduler_decay
        )
        self.adaptation_steps = kwargs.get(
            "adaptation_steps", LightningMAML.adaptation_steps
        )
        self.adaptation_lr = kwargs.get("adaptation_lr", LightningMAML.adaptation_lr)
        self.data_parallel = kwargs.get("data_parallel", False)
        self.features = features
        if self.data_parallel and torch.cuda.device_count() > 1:
            self.features = torch.nn.DataParallel(self.features)
        self.classifier = l2l.algorithms.MAML(classifier, lr=self.adaptation_lr)
        self.save_hyperparameters({
            "train_ways": self.train_ways,
            "train_shots": self.train_shots,
            "train_queries": self.train_queries,
            "test_ways": self.test_ways,
            "test_shots": self.test_shots,
            "test_queries": self.test_queries,
            "lr": self.lr,
            "scheduler_step": self.scheduler_step,
            "scheduler_decay": self.scheduler_decay,
            "adaptation_lr": self.adaptation_lr,
            "adaptation_steps": self.adaptation_steps,
        })
        assert (
            self.train_ways == self.test_ways
        ), "For ANIL, train_ways should be equal to test_ways."

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = LightningEpisodicModule.add_model_specific_args(parent_parser)
        parser.add_argument(
            "--adaptation_steps",
            type=int,
            default=LightningMAML.adaptation_steps,
        )
        parser.add_argument(
            "--adaptation_lr",
            type=float,
            default=LightningMAML.adaptation_lr,
        )
        parser.add_argument(
            "--data_parallel",
            action='store_true',
            help='Use this + CUDA_VISIBLE_DEVICES to parallelize across GPUs.',
        )
        return parser

    @torch.enable_grad()
    def meta_learn(self, batch, batch_idx, ways, shots, queries):
        self.features.train()
        learner = self.classifier.clone()
        learner.train()
        data, labels = batch
        data = self.features(data)

        # Separate data into adaptation and evaluation sets
        support_indices = np.zeros(data.size(0), dtype=bool)
        selection = np.arange(ways) * (shots + queries)
        for offset in range(shots):
            support_indices[selection + offset] = True
        query_indices = torch.from_numpy(~support_indices)
        support_indices = torch.from_numpy(support_indices)
        support = data[support_indices]
        support_labels = labels[support_indices]
        query = data[query_indices]
        query_labels = labels[query_indices]

        # Adapt the classifier
        for step in range(self.adaptation_steps):
            preds = learner(support)
            train_error = self.loss(preds, support_labels)
            learner.adapt(train_error)

        # Evaluating the adapted model
        predictions = learner(query)
        valid_error = self.loss(predictions, query_labels)
        valid_accuracy = accuracy(predictions, query_labels)
        return valid_error, valid_accuracy
