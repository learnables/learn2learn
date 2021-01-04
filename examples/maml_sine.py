#!/usr/bin/env python3

"""
Trains a 3 layer MLP with MAML on Sine Wave Regression Dataset. 
We use the Sine Wave dataloader from the torchmeta package.

Torchmeta: https://github.com/tristandeleu/pytorch-meta
"""

import random
import numpy as np
import torch
import learn2learn as l2l

from torch import nn, optim

from torchmeta.toy import Sinusoid
from torchmeta.utils.data import BatchMetaDataLoader



class SineModel(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.hidden1 = nn.Linear(1, dim)
        self.hidden2 = nn.Linear(dim, dim)
        self.hidden3 = nn.Linear(dim, 1)
        
    def forward(self, x):
        x = nn.functional.relu(self.hidden1(x))
        x = nn.functional.relu(self.hidden2(x))
        x = self.hidden3(x)
        return x


def main(
        shots=10,
        tasks_per_batch=16,
        num_tasks=160000,
        adapt_lr=0.01,
        meta_lr=0.001,
        adapt_steps=5,
        hidden_dim=32,
):
    # load the dataset
    tasksets = Sinusoid(num_samples_per_task=2*shots, num_tasks=num_tasks)
    dataloader = BatchMetaDataLoader(tasksets, batch_size=tasks_per_batch)

    # create the model
    model = SineModel(dim=hidden_dim)
    maml = l2l.algorithms.MAML(model, lr=adapt_lr, first_order=False, allow_unused=True)
    opt = optim.Adam(maml.parameters(), meta_lr)
    lossfn = nn.MSELoss(reduction='mean')

    # for each iteration
    for iter, batch in enumerate(dataloader): # num_tasks/batch_size
        meta_train_loss = 0.0

        # for each task in the batch
        effective_batch_size = batch[0].shape[0]
        for i in range(effective_batch_size):
            learner = maml.clone()

            # divide the data into support and query sets
            train_inputs, train_targets = batch[0][i].float(), batch[1][i].float()
            x_support, y_support = train_inputs[::2], train_targets[::2]
            x_query, y_query = train_inputs[1::2], train_targets[1::2]


            for _ in range(adapt_steps): # adaptation_steps
                support_preds = learner(x_support)
                support_loss=lossfn(support_preds, y_support)
                learner.adapt(support_loss)

            query_preds = learner(x_query)
            query_loss = lossfn(query_preds, y_query)
            meta_train_loss += query_loss

        meta_train_loss = meta_train_loss / effective_batch_size

        if iter % 200 == 0:
            print('Iteration:', iter, 'Meta Train Loss', meta_train_loss.item()) 

        opt.zero_grad()
        meta_train_loss.backward()
        opt.step()

if __name__ == '__main__':
    main()
