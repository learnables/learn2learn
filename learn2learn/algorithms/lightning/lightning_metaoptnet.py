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
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/algorithms/lightning/lightning_metaoptnet.py)

    **Description**

    A PyTorch Lightning module for MetaOptNet.

    **Arguments**

    * **features** (Module) - Feature extractor which classifies input tasks.
    * **svm_C_reg** (float, *optional*, default=0.1) - Regularization weight for SVM.
    * **svm_max_iters** (int, *optional*, default=15) - Maximum number of iterations for SVM convergence.
    * **loss** (Function, *optional*, default=CrossEntropyLoss) - Loss function which maps the cost of the events.
    * **train_ways** (int, *optional*, default=5) - Number of classes in for train tasks.
    * **train_shots** (int, *optional*, default=1) - Number of support samples for train tasks.
    * **train_queries** (int, *optional*, default=1) - Number of query samples for train tasks.
    * **test_ways** (int, *optional*, default=5) - Number of classes in for test tasks.
    * **test_shots** (int, *optional*, default=1) - Number of support samples for test tasks.
    * **test_queries** (int, *optional*, default=1) - Number of query samples for test tasks.
    * **lr** (float, *optional*, default=0.001) - Learning rate of meta training.
    * **scheduler_step** (int, *optional*, default=20) - Decay interval for `lr`.
    * **scheduler_decay** (float, *optional*, default=1.0) - Decay rate for `lr`.

    **References**

    1. Lee et al. 2019. "Meta-Learning with Differentiable Convex Optimization"

    **Example**

    ~~~python
    tasksets = l2l.vision.benchmarks.get_tasksets('mini-imagenet')
    features = Convnet()  # init model
    metaoptnet = LightningMetaOptNet(features, **dict_args)
    episodic_data = EpisodicBatcher(tasksets.train, tasksets.validation, tasksets.test)
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(metaoptnet, episodic_data)
    ~~~
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
