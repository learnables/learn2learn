#!/usr/bin/env python3

"""
Trains MAML using PG + Baseline + GAE for fast adaptation,
and A2C for meta-learning.
"""

import gym
import random
import numpy as np
import torch as th
import cherry as ch
import learn2learn as l2l

from torch import autograd, optim
from cherry.algorithms import a2c
from cherry.models.control import Actor

from policies import DiagNormalPolicy, LinearValue

ENV_SIZE = 11
START = (int((ENV_SIZE+1)/2), int((ENV_SIZE+1)/2))
GOAL = (1, 1)

def maml_a2c_loss(train_episodes, learner, baseline, gamma, tau):
    # Update policy and baseline
    rewards = train_episodes.reward()
    states = train_episodes.state()
    densities = learner(states)[1]['density']
    log_probs = densities.log_prob(train_episodes.action())
    log_probs = log_probs.mean(dim=1, keepdim=True)
    dones = train_episodes.done()

    # Update baseline
    returns = ch.td.discount(gamma, rewards, dones)
    baseline.fit(states, returns)
    values = baseline(states)

    # Update model
    advantages = ch.pg.generalized_advantage(tau=tau,
                                             gamma=gamma,
                                             rewards=rewards,
                                             dones=dones,
                                             values=values,
                                             next_value=th.zeros(1))
    return a2c.policy_loss(log_probs, advantages)


def main(
        experiment='dev',
        task_name='nav2d',
        adapt_lr=0.1,
        meta_lr=0.01,
        adapt_steps=1,
        num_iterations=200,
        meta_bsz=4,
        adapt_bsz=32,
        tau=1.00,
        gamma=0.99,
        seed=42,
        ):

    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)

    if task_name == 'nav2d':
        env_name = '2DNavigation-v0'

    env_name ='MiniGrid-Empty-v0'
    env = gym.make(env_name, size=ENV_SIZE)
    env.seed(seed)
    env = ch.envs.Torch(env)
    policy = Actor(env.state_size, env.action_size)
    maml = l2l.MetaSGD(policy, lr=meta_lr)
    baseline = LinearValue(env.state_size, env.action_size)
    opt = optim.Adam(policy.parameters(), lr=meta_lr)

    for iteration in range(num_iterations):
        iteration_loss = 0.0
        iteration_reward = 0.0
        for task_config in env.sample_tasks(meta_bsz):  # Samples a new config
            learner = maml.new()
            env.reset_task(task_config)
            env.reset()
            task = ch.envs.Runner(env)

            # Fast Adapt
            for step in range(adapt_steps):
                train_episodes = task.run(learner, episodes=adapt_bsz)
                loss = maml_a2c_loss(train_episodes, learner, baseline, gamma, tau)
                learner.adapt(loss)

            # Compute Validation Loss
            valid_episodes = task.run(learner, episodes=adapt_bsz)
            loss = maml_a2c_loss(valid_episodes, learner, baseline, gamma, tau)
            iteration_loss += loss
            iteration_reward += valid_episodes.reward().sum().item() / adapt_bsz

        opt.zero_grad()
        iteration_loss.backward()
        opt.step()

        # Print statistics
        print('\nIteration', iteration)
        adaptation_reward = iteration_reward / meta_bsz
        print('adaptation_reward', adaptation_reward)

        adaptation_loss = iteration_loss.item() / meta_bsz
        print('adaptation_loss', adaptation_loss)


if __name__ == '__main__':
    main()
