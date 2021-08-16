#!/usr/bin/env python3

import numpy as np
import torch
import learn2learn as l2l

from learn2learn.utils import accuracy
from learn2learn.algorithms.lightning import LightningEpisodicModule


class LightningMAML(LightningEpisodicModule):
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/algorithms/lightning/lightning_maml.py)

    **Description**

    A PyTorch Lightning module for MAML.

    **Arguments**

    * **model** (Module) - A PyTorch nn.Module.
    * **loss** (Function, *optional*, default=CrossEntropyLoss) - Loss function which maps the cost of the events.
    * **ways** (int, *optional*, default=5) - Number of classes in a task.
    * **shots** (int, *optional*, default=1) - Number of samples for adaptation.
    * **adaptation_steps** (int, *optional*, default=1) - Number of steps for adapting to new task.
    * **lr** (float, *optional*, default=0.001) - Learning rate of meta training.
    * **adaptation_lr** (float, *optional*, default=0.1) - Learning rate for fast adaptation.
    * **scheduler_step** (int, *optional*, default=20) - Decay interval for `lr`.
    * **scheduler_decay** (float, *optional*, default=1.0) - Decay rate for `lr`.

    **References**

    1. Finn et al. 2017. "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks."

    **Example**

    ~~~python
    tasksets = l2l.vision.benchmarks.get_tasksets('omniglot')
    model = l2l.vision.models.OmniglotFC(28**2, args.ways)
    maml = LightningMAML(classifier, adaptation_lr=0.1, **dict_args)
    episodic_data = EpisodicBatcher(tasksets.train, tasksets.validation, tasksets.test)
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(maml, episodic_data)
    ~~~
    """

    adaptation_steps = 1
    adaptation_lr = 0.1

    def __init__(self, model, loss=None, **kwargs):
        super(LightningMAML, self).__init__()
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
        self.data_parallel = kwargs.get("data_parallel", False) and torch.cuda.device_count() > 1
        assert (
            self.train_ways == self.test_ways
        ), "For MAML, train_ways should be equal to test_ways."
        self.model = l2l.algorithms.MAML(model, lr=self.adaptation_lr, first_order=False)

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
        learner = self.model.clone()
        learner.train()
        if self.data_parallel:
            learner.module = torch.nn.DataParallel(learner.module)
        data, labels = batch

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

        # Adapt the model
        for step in range(self.adaptation_steps):
            train_error = self.loss(learner(support), support_labels)
            learner.adapt(train_error)

        # Evaluating the adapted model
        predictions = learner(query)
        valid_error = self.loss(predictions, query_labels)
        valid_accuracy = accuracy(predictions, query_labels)
        return valid_error, valid_accuracy
