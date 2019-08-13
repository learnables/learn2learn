#!/usr/bin/env python3

"""
Trains MAML using PG + Baseline + GAE for fast adaptation,
and PPO for meta-learning.
"""

import random
from copy import deepcopy

import cherry as ch
import gym
import numpy as np
import randopt as ro
import torch as th
from cherry.algorithms import a2c, ppo
from policies import DiagNormalPolicy, LinearValue
from torch import autograd, optim
from tqdm import tqdm

import learn2learn as l2l

import wandb
wandb.init(project="learn2learn")

def compute_advantages(baseline, tau, gamma, rewards, dones, states):
    # Update baseline
    returns = ch.td.discount(gamma, rewards, dones)
    baseline.fit(states, returns)
    values = baseline(states)
    next_value = th.zeros(1, device=values.device)
    return ch.pg.generalized_advantage(tau=tau,
                                       gamma=gamma,
                                       rewards=rewards,
                                       dones=dones,
                                       values=values,
                                       next_value=next_value)


def maml_a2c_loss(train_episodes, clone, baseline, gamma, tau):
    # Update policy and baseline
    rewards = train_episodes.reward()
    states = train_episodes.state()
    densities = clone(states)[1]['density']
    log_probs = densities.log_prob(train_episodes.action())
    log_probs = log_probs.mean(dim=1, keepdim=True)
    dones = train_episodes.done()
    advantages = compute_advantages(baseline, tau, gamma, rewards, dones, states)
    return a2c.policy_loss(log_probs, advantages)


def fast_adapt_a2c(clone, train_episodes, adapt_lr, baseline, gamma, tau, first_order=False):
    second_order = not first_order
    loss = maml_a2c_loss(train_episodes, clone, baseline, gamma, tau)
    gradients = autograd.grad(loss,
                              clone.parameters(),
                              retain_graph=second_order,
                              create_graph=second_order)
    return l2l.maml.maml_update(clone, adapt_lr, gradients)


def main(
        experiment='dev',
        task_name='nav2d',
        adapt_lr=0.1,
        meta_lr=1e-4,
        adapt_steps=1,
        num_iterations=20,
        meta_bsz=40,
        adapt_bsz=20,
        tau=1.00,
        gamma=0.99,
        seed=4210,
):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)

    if task_name == 'nav2d':
        env_name = '2DNavigation-v0'

    env = gym.make(env_name)
    env.seed(seed)
    env = ch.envs.Torch(env)
    policy = DiagNormalPolicy(env.state_size, env.action_size)

    baseline = LinearValue(env.state_size, env.action_size)
    opt = optim.Adam(policy.parameters(), lr=meta_lr, eps=1e-5)

    all_rewards = []
    for iteration in range(num_iterations):
        iteration_reward = 0.0
        iteration_replays = []
        policy.to('cpu')
        baseline.to('cpu')

        for task_config in tqdm(env.sample_tasks(meta_bsz), leave=False, desc='Data'):  # Samples a new config
            clone = deepcopy(policy)
            env.reset_task(task_config)
            env.reset()
            task = ch.envs.Runner(env)
            task_replay = []

            # Fast Adapt
            for step in range(adapt_steps):
                train_episodes = task.run(clone, episodes=adapt_bsz)
                clone = fast_adapt_a2c(clone, train_episodes, adapt_lr,
                                       baseline, gamma, tau, first_order=True)
                task_replay.append(train_episodes)

            # Compute Validation Loss
            valid_episodes = task.run(clone, episodes=adapt_bsz)
            task_replay.append(valid_episodes)
            iteration_reward += valid_episodes.reward().sum().item() / adapt_bsz
            iteration_replays.append(task_replay)

        # Print statistics
        print('\nIteration', iteration)
        adaptation_reward = iteration_reward / meta_bsz
        all_rewards.append(adaptation_reward)
        wandb.log({'PPO - Adaptation Reward': adaptation_reward})
        print('adaptation_reward', adaptation_reward)

        # PPO meta-optimization
        for ppo_step in tqdm(range(5), leave=False, desc='Optim'):
            ppo_loss = 0.0
            for task_replays in iteration_replays:
                train_replays = task_replays[:-1]
                valid_replay = task_replays[-1]

                # Fast adapt clone, starting from the current init
                clone = l2l.maml.clone_module(policy)
                for train_episodes in train_replays:
                    clone = fast_adapt_a2c(clone, train_episodes, adapt_lr,
                                           baseline, gamma, tau, first_order=False)

                # Compute PPO loss between old and new clones
                states = valid_replay.state()
                actions = valid_replay.action()
                rewards = valid_replay.reward()
                dones = valid_replay.done()
                old_log_probs = valid_replay.log_prob()
                new_densities = clone(states)[1]['density']
                new_log_probs = new_densities.log_prob(actions).mean(dim=1, keepdim=True)
                advantages = compute_advantages(baseline, tau, gamma, rewards, dones, states)
                ppo_loss += ppo.policy_loss(new_log_probs, old_log_probs, advantages, clip=0.1)
            ppo_loss /= meta_bsz
            opt.zero_grad()
            ppo_loss.backward()
            opt.step()
    th.save(all_rewards, 'ppo.data')


if __name__ == '__main__':
    main()
