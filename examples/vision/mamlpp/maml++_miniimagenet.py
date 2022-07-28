#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Example implementation of MAML++ on miniImageNet.
"""


import learn2learn as l2l
import numpy as np
import random
import torch

from collections import namedtuple
from typing import Tuple
from tqdm import tqdm
from learn2learn.optim.transforms.layer_step_lr_transform import PerLayerPerStepLRTransform

from learn2learn.vision.models.cnn4 import CNN4


MetaBatch = namedtuple("MetaBatch", "support query")

train_samples, val_samples, test_samples = 38400, 9600, 12000  # Is that correct?
tasks = 600


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


class MAMLppTrainer:
    def __init__(
        self,
        ways=20,
        k_shots=5,
        n_queries=5,
        steps=5,
        msl_epochs=25,
        DA_epochs=50,
        use_cuda=True,
        seed=42,
    ):
        self._use_cuda = use_cuda
        self._device = torch.device("cpu")
        if self._use_cuda and torch.cuda.device_count():
            torch.cuda.manual_seed(seed)
            self._device = torch.device("cuda")
        print(f"[*] Using device: {self._device}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Dataset
        print("[*] Loading mini-ImageNet...")
        (
            self._train_tasks,
            self._valid_tasks,
            self._test_tasks,
        ) = l2l.vision.benchmarks.get_tasksets(
            "mini-imagenet",
            train_samples=k_shots+n_queries,
            train_ways=ways,
            test_samples=k_shots+n_queries,
            test_ways=ways,
            root="~/data",
        )
        print("[*] Done.")

        # Model
        self._model = CNN4(ways) # TODO: Change config for miniImageNet (32 filters ?)
        if self._use_cuda:
            self._model.cuda()

        # Meta-Learning related
        self._steps = steps
        self._k_shots = k_shots
        self._n_queries = n_queries
        self._inner_criterion = torch.nn.CrossEntropyLoss(reduction="mean")

        # Multi-Step Loss
        self._msl_epochs = msl_epochs
        self._step_weights = torch.ones(steps, device=self._device) * (1.0 / steps)
        self._msl_decay_rate = 1.0 / steps / msl_epochs
        self._msl_min_value_for_non_final_losses = torch.tensor(0.03 / steps)
        self._msl_max_value_for_final_loss = 1.0 - (
            (steps - 1) * self._msl_min_value_for_non_final_losses
        )

        # Derivative-Order Annealing (when to start using second-order opt)
        self._derivative_order_annealing_from_epoch = DA_epochs

    def _anneal_step_weights(self):
        self._step_weights[:-1] = torch.max(
            self._step_weights[:-1] - self._msl_decay_rate,
            self._msl_min_value_for_non_final_losses,
        )
        self._step_weights[-1] = torch.min(
            self._step_weights[-1] + ((self._steps - 1) * self._msl_decay_rate),
            self._msl_max_value_for_final_loss,
        )

    def _split_batch(self, batch: tuple) -> MetaBatch:
        """
        Separate data batch into adaptation/evalutation sets.
        """
        images, labels = batch
        task_size = self._k_shots + self._n_queries
        assert task_size <= images.shape[0], "K+N are smaller than the batch size!"
        indices = torch.randperm(task_size)
        support_indices = indices[: self._k_shots]
        query_indices = indices[self._k_shots :]
        return MetaBatch(
            (
                images[support_indices],
                labels[support_indices],
            ),
            (images[query_indices], labels[query_indices]),
        )

    def _training_step(
        self,
        batch: MetaBatch,
        learner: torch.nn.Module,
        msl: bool = True,
        epoch: int = 0,
    ) -> Tuple[torch.Tensor, float]:
        s_inputs, s_labels = batch.support
        q_inputs, q_labels = batch.query
        query_loss = torch.tensor(.0, device=self._device)

        if self._use_cuda:
            s_inputs = s_inputs.float().cuda(device=self._device)
            s_labels = s_labels.cuda(device=self._device)
            q_inputs = q_inputs.float().cuda(device=self._device)
            q_labels = q_labels.cuda(device=self._device)

        # Derivative-Order Annealing
        second_order = True
        if epoch < self._derivative_order_annealing_from_epoch:
            second_order = False

        # Adapt the model on the support set
        for step in range(self._steps):
            # forward + backward + optimize
            pred = learner(s_inputs)
            support_loss = self._inner_criterion(pred, s_labels)
            learner.adapt(support_loss, first_order=not second_order)
            # Multi-Step Loss
            if msl:
                q_pred = learner(q_inputs)
                query_loss += self._step_weights[step] * self._inner_criterion(
                    q_pred, q_labels
                )

        # Evaluate the adapted model on the query set
        if not msl:
            q_pred = learner(q_inputs, inference=True)
            query_loss = self._inner_criterion(q_pred, q_labels)
        acc = accuracy(q_pred, q_labels).detach()

        return query_loss, acc

    def _testing_step(
        self, batch: MetaBatch, learner: torch.nn.Module
    ) -> Tuple[torch.Tensor, float]:
        s_inputs, s_labels = batch.support
        q_inputs, q_labels = batch.query

        if self._use_cuda:
            s_inputs = s_inputs.float().cuda(device=self._device)
            s_labels = s_labels.cuda(device=self._device)
            q_inputs = q_inputs.float().cuda(device=self._device)
            q_labels = q_labels.cuda(device=self._device)

        # Adapt the model on the support set
        for step in range(self._steps):
            # forward + backward + optimize
            pred = learner(s_inputs)
            support_loss = self._inner_criterion(pred, s_labels)
            learner.adapt(support_loss)

        # Evaluate the adapted model on the query set
        q_pred = learner(q_inputs, inference=True)
        query_loss = self._inner_criterion(q_pred, q_labels).detach()
        acc = accuracy(q_pred, q_labels)

        return query_loss, acc

    def train(
        self,
        meta_lr=0.001,
        fast_lr=0.01,
        meta_bsz=16,
        epochs=1,
        val_interval=1,
    ):
        print("[*] Training...")
        transform = PerLayerPerStepLRTransform(fast_lr, self._steps, self._model, ["conv"])
        # Setting adapt_transform=True means that the transform will be updated in
        # the *adapt* function, which is not what we want. We want it to compute gradients during
        # eval_loss.backward() only, so that it's updated in opt.step().
        mamlpp = l2l.algorithms.GBML(
            self._model,
            transform,
            lr=1.0,
            allow_nograd=True,
            adapt_transform=False,
            pass_param_names=True,
        )
        opt = torch.optim.AdamW(mamlpp.parameters(), meta_lr, betas=(0.9, 0.99))

        iter_per_epoch = (
            train_samples // (meta_bsz * (self._k_shots + self._n_queries))
        ) + 1
        print(f"[*] Training with {iter_per_epoch} iterations/epoch with {train_samples} total training samples")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=epochs * iter_per_epoch,
            eta_min=0.00001,
        )

        for epoch in range(epochs):
            epoch_meta_train_loss, epoch_meta_train_acc = 0.0, 0.0
            for _ in tqdm(range(iter_per_epoch)):
                opt.zero_grad()
                meta_train_losses, meta_train_accs = [], []

                for _ in range(meta_bsz):
                    meta_batch = self._split_batch(self._train_tasks.sample())
                    meta_loss, meta_acc = self._training_step(
                        meta_batch,
                        mamlpp.clone(),
                        msl=(epoch < self._msl_epochs),
                        epoch=epoch,
                    )
                    meta_loss.backward()
                    meta_train_losses.append(meta_loss.detach())
                    meta_train_accs.append(meta_acc)

                epoch_meta_train_loss += torch.Tensor(meta_train_losses).mean().item()
                epoch_meta_train_acc += torch.Tensor(meta_train_accs).mean().item()

                # Average the accumulated gradients and optimize
                with torch.no_grad():
                    for p in mamlpp.parameters():
                        # Remember the MetaBatchNorm layer has parameters that don't require grad!
                        if p.requires_grad:
                            p.grad.data.mul_(1.0 / meta_bsz)

                opt.step()
                scheduler.step()
                # Multi-Step Loss
                self._anneal_step_weights()

            epoch_meta_train_loss /= iter_per_epoch
            epoch_meta_train_acc /= iter_per_epoch
            print(f"==========[Epoch {epoch}]==========")
            print(f"Meta-training Loss: {epoch_meta_train_loss:.6f}")
            print(f"Meta-training Acc: {epoch_meta_train_acc:.6f}")

            # ======= Validation ========
            if (epoch + 1) % val_interval == 0:
                # Compute the meta-validation loss
                # TODO: Go through the entire validation set, which shouldn't be shuffled, and
                # which tasks should not be continuously resampled from!
                # This may be done in the get_tasksets() method actually...
                meta_val_losses, meta_val_accs = [], []
                for _ in tqdm(range(val_samples // tasks)):
                    meta_batch = self._split_batch(self._valid_tasks.sample())
                    loss, acc = self._testing_step(meta_batch, mamlpp.clone())
                    meta_val_losses.append(loss)
                    meta_val_accs.append(acc)
                meta_val_loss = float(torch.Tensor(meta_val_losses).mean().item())
                meta_val_acc = float(torch.Tensor(meta_val_accs).mean().item())
                print(f"Meta-validation Loss: {meta_val_loss:.6f}")
                print(f"Meta-validation Accuracy: {meta_val_acc:.6f}")
            print("============================================")

        return self._model.state_dict(), transform.state_dict()

    def test(
        self,
        model_state_dict,
        trasnform_state_dict,
        meta_lr=0.001,
        fast_lr=0.01,
        meta_bsz=5,
    ):
        self._model.load_state_dict(model_state_dict)
        transform = PerLayerPerStepLRTransform(fast_lr, self._steps, self._model, ["conv"])
        transform.load_state_dict(trasnform_state_dict)
        # Setting adapt_transform=True means that the transform will be updated in
        # the *adapt* function, which is not what we want. We want it to compute gradients during
        # eval_loss.backward() only, so that it's updated in opt.step().
        mamlpp = l2l.algorithms.GBML(
            self._model,
            transform,
            lr=1.0,
            adapt_transform=False,
            pass_param_names=True,
        )

        meta_losses, meta_accs = [], []
        for _ in tqdm(range(test_samples // tasks)):
            meta_batch = self._split_batch(self._test_tasks.sample())
            loss, acc = self._testing_step(meta_batch, mamlpp.clone())
            meta_losses.append(loss)
            meta_accs.append(acc)
        loss = float(torch.Tensor(meta_losses).mean().item())
        acc = float(torch.Tensor(meta_accs).mean().item())
        print(f"Meta-training Loss: {loss:.6f}")
        print(f"Meta-training Acc: {acc:.6f}")


if __name__ == "__main__":
    mamlPlusPlus = MAMLppTrainer()
    model_state_dict, transform_state_dict = mamlPlusPlus.train()
    mamlPlusPlus.test(model_state_dict, transform_state_dict)

