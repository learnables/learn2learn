#!/usr/bin/env python3

"""
Trains MAML using PG + Baseline + GAE for fast adaptation,
and A2C for meta-learning.
"""

import random
import gym
import numpy as np
import torch as th
import cherry as ch

import learn2learn as l2l

from torch import optim
from cherry.algorithms import a2c
from tqdm import tqdm

from policies import DiagNormalPolicy, LinearValue


def compute_advantages(baseline, tau, gamma, rewards, dones, states, next_states):
    # Update baseline
    returns = ch.td.discount(gamma, rewards, dones)
    baseline.fit(states, returns)
    values = baseline(states)
    next_values = baseline(next_states)
    bootstraps = values * (1.0 - dones) + next_values * dones
    next_value = th.zeros(1, device=values.device)
    return ch.pg.generalized_advantage(tau=tau,
                                       gamma=gamma,
                                       rewards=rewards,
                                       dones=dones,
                                       values=bootstraps,
                                       next_value=next_value)


def maml_a2c_loss(train_episodes, learner, baseline, gamma, tau):
    # Update policy and baseline
    states = train_episodes.state()
    actions = train_episodes.action()
    rewards = train_episodes.reward()
    dones = train_episodes.done()
    next_states = train_episodes.next_state()
    log_probs = learner.log_prob(states, actions)
    advantages = compute_advantages(baseline, tau, gamma, rewards,
                                    dones, states, next_states)
    advantages = ch.normalize(advantages).detach()
    return a2c.policy_loss(log_probs, advantages)


def main(
        experiment='dev',
        task_name='nav2d',
        adapt_lr=0.1,
        meta_lr=0.01,
        adapt_steps=1,
        num_iterations=200,
        meta_bsz=20,
        adapt_bsz=20,
        tau=1.00,
        gamma=0.99,
        num_workers=8,
        seed=42,
):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)

    if task_name == 'nav2d':
        env_name = '2DNavigation-v0'

    def make_env():
        return gym.make(env_name)

    env = l2l.gym.AsyncVectorEnv([make_env for _ in range(num_workers)])
    env.seed(seed)
    env = ch.envs.Torch(env)
    policy = DiagNormalPolicy(env.state_size, env.action_size)
    meta_learner = l2l.MAML(policy, lr=meta_lr)
    baseline = LinearValue(env.state_size, env.action_size)
    opt = optim.Adam(policy.parameters(), lr=meta_lr)
    all_rewards = []

    for iteration in range(num_iterations):
        iteration_loss = 0.0
        iteration_reward = 0.0
        for task_config in tqdm(env.sample_tasks(meta_bsz)):  # Samples a new config
            learner = meta_learner.new()
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

        # Print statistics
        print('\nIteration', iteration)
        adaptation_reward = iteration_reward / meta_bsz
        print('adaptation_reward', adaptation_reward)
        all_rewards.append(adaptation_reward)

        adaptation_loss = iteration_loss / meta_bsz
        print('adaptation_loss', adaptation_loss.item())

        opt.zero_grad()
        adaptation_loss.backward()
        opt.step()


if __name__ == '__main__':
    main()
