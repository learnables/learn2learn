#!/usr/bin/env python3

"""
This script demonstrates how to use the MAML implementation of L2L.

Each task i consists of learning the parameters of a Normal distribution N(mu_i, sigma_i).
The parameters mu_i, sigma_i are themselves sampled from a distribution N(mu, sigma).
"""

import torch as th
from torch import nn, optim, distributions as dist

import learn2learn as l2l

DIM = 5
TIMESTEPS = 1000
TASKS_PER_STEP = 50


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.mu = nn.Parameter(th.randn(DIM))
        self.sigma = nn.Parameter(th.randn(DIM))

    def forward(self, x=None):
        return dist.Normal(self.mu, self.sigma)


def main():
    task_dist = dist.Normal(th.zeros(2 * DIM), th.ones(2 * DIM))
    model = Model()
    maml = l2l.algorithms.MAML(model, lr=1e-2)
    opt = optim.Adam(maml.parameters())

    for i in range(TIMESTEPS):
        step_loss = 0.0
        for t in range(TASKS_PER_STEP):
            # Sample a task
            task_params = task_dist.sample()
            mu_i, sigma_i = task_params[:DIM], task_params[DIM:]

            # Adaptation: Instanciate a copy of model
            learner = maml.clone()
            proposal = learner()

            # Adaptation: Compute and adapt to task loss
            loss = (mu_i - proposal.mean).pow(2).sum() + (sigma_i - proposal.variance).pow(2).sum()
            learner.adapt(loss)

            # Adaptation: Evaluate the effectiveness of adaptation
            adapt_loss = (mu_i - proposal.mean).pow(2).sum() + (sigma_i - proposal.variance).pow(2).sum()

            # Accumulate the error over all tasks
            step_loss += adapt_loss

        # Meta-learning step: compute gradient through the adaptation step, automatically.
        step_loss = step_loss / TASKS_PER_STEP
        print(i, step_loss.item())
        opt.zero_grad()
        step_loss.backward()
        opt.step()


if __name__ == '__main__':
    main()
