#!/usr/bin/env python3

import numpy as np
import torch

from learn2learn.utils import accuracy
from learn2learn.nn import SVClassifier
from learn2learn.algorithms.lightning import (
    LightningEpisodicModule,
    LightningPrototypicalNetworks,
)


class LightningMetaOptNet(LightningPrototypicalNetworks):
    """
    """

    svm_C_reg = 0.1
    svm_max_iters = 15

    def __init__(self, features, loss=None, **kwargs):
        super(LightningMetaOptNet, self).__init__(features=features, **kwargs)
        self.svm_C_reg = kwargs.get("svm_C_reg", LightningMetaOptNet.svm_C_reg)
        self.svm_max_iters = kwargs.get(
            "svm_max_iters", LightningMetaOptNet.svm_max_iters
        )
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
            "svm_C_reg": self.svm_C_reg,
            "svm_max_iters": self.svm_max_iters,
        })
        self.classifier = SVClassifier(
            C_reg=self.svm_C_reg,
            max_iters=self.svm_max_iters,
            normalize=False,
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = LightningEpisodicModule.add_model_specific_args(parent_parser)
        parser.add_argument(
            "--svm_C_reg",
            type=float,
            default=LightningMetaOptNet.svm_C_reg,
        )
        parser.add_argument(
            "--svm_max_iters",
            type=int,
            default=LightningMetaOptNet.svm_max_iters,
        )
        parser.add_argument(
            "--data_parallel",
            action='store_true',
            help='Use this + CUDA_VISIBLE_DEVICES to parallelize across GPUs.',
        )
        return parser

    def meta_learn(self, batch, batch_idx, ways, shots, queries):
        self.features.train()
        data, labels = batch

        # Sort data samples by labels
        sort = torch.sort(labels)
        data = data.squeeze(0)[sort.indices].squeeze(0)
        labels = labels.squeeze(0)[sort.indices].squeeze(0)

        # Compute support and query embeddings
        embeddings = self.features(data)
        support_indices = np.zeros(data.size(0), dtype=bool)
        selection = np.arange(ways) * (shots + queries)
        for offset in range(shots):
            support_indices[selection + offset] = True
        query_indices = torch.from_numpy(~support_indices)
        support_indices = torch.from_numpy(support_indices)
        support = embeddings[support_indices]
        support_labels = labels[support_indices]
        query = embeddings[query_indices]
        query_labels = labels[query_indices]

        self.classifier.fit_(support, support_labels, ways=ways)
        logits = self.classifier(query)
        eval_loss = self.loss(logits, query_labels)
        eval_accuracy = accuracy(logits, query_labels)
        return eval_loss, eval_accuracy
