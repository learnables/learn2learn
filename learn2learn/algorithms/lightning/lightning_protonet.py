#!/usr/bin/env python3

"""
"""
import numpy as np
import torch

from torch import nn
from learn2learn.utils import accuracy
from learn2learn.nn import PrototypicalClassifier
from learn2learn.algorithms.lightning import LightningEpisodicModule


class LightningPrototypicalNetworks(LightningEpisodicModule):
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/algorithms/lightning/lightning_protonet.py)

    **Description**

    A PyTorch Lightning module for Prototypical Networks.

    **Arguments**

    * **features** (Module) - Feature extractor which classifies input tasks.
    * **loss** (Function, *optional*, default=CrossEntropyLoss) - Loss function which maps the cost of the events.
    * **distance_metric** (str, *optional*, default='euclidean') - Distance metric between samples. ['euclidean', 'cosine']
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

    1. Snell et al. 2017. "Prototypical Networks for Few-shot Learning"

    **Example**

    ~~~python
    tasksets = l2l.vision.benchmarks.get_tasksets('mini-imagenet')
    features = Convnet()  # init model
    protonet = LightningPrototypicalNetworks(features, **dict_args)
    episodic_data = EpisodicBatcher(tasksets.train, tasksets.validation, tasksets.test)
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(protonet, episodic_data)
    ~~~
    """

    distance_metric = "euclidean"

    def __init__(self, features, loss=None, **kwargs):
        super(LightningPrototypicalNetworks, self).__init__()
        if loss is None:
            loss = nn.CrossEntropyLoss(reduction="mean")
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
        self.distance_metric = kwargs.get(
            "distance_metric", LightningPrototypicalNetworks.distance_metric
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
            "distance_metric": self.distance_metric,
        })
        self.data_parallel = kwargs.get("data_parallel", False)
        self.features = features
        if self.data_parallel and torch.cuda.device_count() > 1:
            self.features = torch.nn.DataParallel(self.features)
        self.classifier = PrototypicalClassifier(distance=self.distance_metric)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = LightningEpisodicModule.add_model_specific_args(parent_parser)
        parser.add_argument(
            "--distance_metric",
            type=str,
            default=LightningPrototypicalNetworks.distance_metric,
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

        self.classifier.fit_(support, support_labels)
        logits = self.classifier(query)
        eval_loss = self.loss(logits, query_labels)
        eval_accuracy = accuracy(logits, query_labels)
        return eval_loss, eval_accuracy
