#!/usr/bin/env python3

"""
Trains MAML using PG + Baseline + GAE for fast adaptation,
and PPO for meta-learning.
"""

import random
import gym
import numpy as np
import torch as th
import cherry as ch

import learn2learn as l2l

from cherry.algorithms import a2c, ppo
from torch import autograd, optim

from copy import deepcopy
from tqdm import tqdm

from meta_a2c import compute_advantages, maml_a2c_loss
from policies import DiagNormalPolicy, LinearValue


def fast_adapt_a2c(clone, train_episodes, adapt_lr, baseline, gamma, tau, first_order=False):
    loss = maml_a2c_loss(train_episodes, clone, baseline, gamma, tau)
    clone.adapt(loss, first_order=first_order)
    return clone


def main(
        experiment='dev',
        task_name='nav2d',
        adapt_lr=0.1,
        meta_lr=0.01,
        adapt_steps=1,
        num_iterations=20,
        meta_bsz=10,
        adapt_bsz=10,
        tau=1.00,
        gamma=0.99,
        num_workers=1,
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
    opt = optim.Adam(meta_learner.parameters(), lr=meta_lr)

    all_rewards = []
    for iteration in range(num_iterations):
        iteration_reward = 0.0
        iteration_replays = []
        iteration_policies = []
        policy.to('cpu')
        baseline.to('cpu')

        for task_config in tqdm(env.sample_tasks(meta_bsz), leave=False, desc='Data'):  # Samples a new config
            learner = meta_learner.new()
            env.reset_task(task_config)
            env.reset()
            task = ch.envs.Runner(env)
            task_replay = []

            # Fast Adapt
            for step in range(adapt_steps):
                train_episodes = task.run(learner, episodes=adapt_bsz)
                learner = fast_adapt_a2c(learner, train_episodes, adapt_lr,
                                       baseline, gamma, tau, first_order=True)
                task_replay.append(train_episodes)

            # Compute Validation Loss
            valid_episodes = task.run(learner, episodes=adapt_bsz)
            task_replay.append(valid_episodes)
            iteration_reward += valid_episodes.reward().sum().item() / adapt_bsz
            iteration_replays.append(task_replay)
            iteration_policies.append(learner)

        # Print statistics
        print('\nIteration', iteration)
        adaptation_reward = iteration_reward / meta_bsz
        all_rewards.append(adaptation_reward)
        print('adaptation_reward', adaptation_reward)

        # PPO meta-optimization
        for ppo_step in tqdm(range(10), leave=False, desc='Optim'):
            ppo_loss = 0.0
            for task_replays, old_policy in zip(iteration_replays, iteration_policies):
                train_replays = task_replays[:-1]
                valid_replay = task_replays[-1]

                # Fast adapt new policy, starting from the current init
                new_policy = meta_learner.new()
                for train_episodes in train_replays:
                    new_policy = fast_adapt_a2c(new_policy, train_episodes, adapt_lr,
                                                baseline, gamma, tau)

                # Compute PPO loss between old and new clones
                states = valid_replay.state()
                actions = valid_replay.action()
                rewards = valid_replay.reward()
                dones = valid_replay.done()
                next_states = valid_replay.next_state()
                old_log_probs = old_policy.log_prob(states, actions).detach()
                new_log_probs = new_policy.log_prob(states, actions).detach()
                advantages = compute_advantages(baseline, tau, gamma, rewards, dones, states, next_states)
                ppo_loss += ppo.policy_loss(new_log_probs, old_log_probs, advantages, clip=0.1)

            ppo_loss /= meta_bsz
            opt.zero_grad()
            ppo_loss.backward()
            opt.step()


if __name__ == '__main__':
    main()
