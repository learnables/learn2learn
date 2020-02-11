#!/usr/bin/env python3

"""
Trains a 2-layer MLP with ProMP.

Usage:

python examples/rl/promp.py
"""

import random
from copy import deepcopy

import cherry as ch
import gym
import numpy as np
import torch
from cherry.algorithms import a2c, ppo, trpo
from cherry.models.robotics import LinearValue
from torch import optim
from torch.distributions.kl import kl_divergence
from tqdm import tqdm

import learn2learn as l2l
from policies import DiagNormalPolicy


def compute_advantages(baseline, tau, gamma, rewards, dones, states, next_states):
    # Update baseline
    returns = ch.td.discount(gamma, rewards, dones)
    baseline.fit(states, returns)
    values = baseline(states)
    next_values = baseline(next_states)
    bootstraps = values * (1.0 - dones) + next_values * dones
    next_value = torch.zeros(1, device=values.device)
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


def fast_adapt_a2c(clone, train_episodes, adapt_lr, baseline, gamma, tau, first_order=False):
    loss = maml_a2c_loss(train_episodes, clone, baseline, gamma, tau)
    clone.adapt(loss, first_order=first_order)
    return clone


def precompute_quantities(states, actions, old_policy, new_policy):
    old_density = old_policy.density(states)
    old_log_probs = old_density.log_prob(actions).mean(dim=1, keepdim=True).detach()
    new_density = new_policy.density(states)
    new_log_probs = new_density.log_prob(actions).mean(dim=1, keepdim=True)
    return old_density, new_density, old_log_probs, new_log_probs


def main(
        env_name='AntDirection-v1',
        adapt_lr=0.1,
        meta_lr=3e-4,
        adapt_steps=3,
        num_iterations=1000,
        meta_bsz=40,
        adapt_bsz=20,
        ppo_clip=0.3,
        ppo_steps=5,
        tau=1.00,
        gamma=0.99,
        eta=0.0005,
        adaptive_penalty=False,
        kl_target=0.01,
        num_workers=4,
        seed=421,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    def make_env():
        env = gym.make(env_name)
        env = ch.envs.ActionSpaceScaler(env)
        return env

    env = l2l.gym.AsyncVectorEnv([make_env for _ in range(num_workers)])
    env.seed(seed)
    env = ch.envs.ActionSpaceScaler(env)
    env = ch.envs.Torch(env)
    policy = DiagNormalPolicy(input_size=env.state_size,
                              output_size=env.action_size,
                              hiddens=[64, 64],
                              activation='tanh')
    meta_learner = l2l.algorithms.MAML(policy, lr=meta_lr)
    baseline = LinearValue(env.state_size, env.action_size)
    opt = optim.Adam(meta_learner.parameters(), lr=meta_lr)

    for iteration in range(num_iterations):
        iteration_reward = 0.0
        iteration_replays = []
        iteration_policies = []

        # Sample Trajectories
        for task_config in tqdm(env.sample_tasks(meta_bsz), leave=False, desc='Data'):
            clone = deepcopy(meta_learner)
            env.set_task(task_config)
            env.reset()
            task = ch.envs.Runner(env)
            task_replay = []
            task_policies = []

            # Fast Adapt
            for step in range(adapt_steps):
                for p in clone.parameters():
                    p.detach_().requires_grad_()
                task_policies.append(deepcopy(clone))
                train_episodes = task.run(clone, episodes=adapt_bsz)
                clone = fast_adapt_a2c(clone, train_episodes, adapt_lr,
                                       baseline, gamma, tau, first_order=True)
                task_replay.append(train_episodes)

            # Compute Validation Loss
            for p in clone.parameters():
                p.detach_().requires_grad_()
            task_policies.append(deepcopy(clone))
            valid_episodes = task.run(clone, episodes=adapt_bsz)
            task_replay.append(valid_episodes)
            iteration_reward += valid_episodes.reward().sum().item() / adapt_bsz
            iteration_replays.append(task_replay)
            iteration_policies.append(task_policies)

        # Print statistics
        print('\nIteration', iteration)
        adaptation_reward = iteration_reward / meta_bsz
        print('adaptation_reward', adaptation_reward)

        # ProMP meta-optimization
        for ppo_step in tqdm(range(ppo_steps), leave=False, desc='Optim'):
            promp_loss = 0.0
            kl_total = 0.0
            for task_replays, old_policies in zip(iteration_replays, iteration_policies):
                new_policy = meta_learner.clone()
                states = task_replays[0].state()
                actions = task_replays[0].action()
                rewards = task_replays[0].reward()
                dones = task_replays[0].done()
                next_states = task_replays[0].next_state()
                old_policy = old_policies[0]
                (old_density,
                 new_density,
                 old_log_probs,
                 new_log_probs) = precompute_quantities(states,
                                                        actions,
                                                        old_policy,
                                                        new_policy)
                advantages = compute_advantages(baseline, tau, gamma, rewards,
                                                dones, states, next_states)
                advantages = ch.normalize(advantages).detach()
                for step in range(adapt_steps):
                    # Compute KL penalty
                    kl_pen = kl_divergence(old_density, new_density).mean()
                    kl_total += kl_pen.item()

                    # Update the clone
                    surr_loss = trpo.policy_loss(new_log_probs, old_log_probs, advantages)
                    new_policy.adapt(surr_loss)

                    # Move to next adaptation step
                    states = task_replays[step + 1].state()
                    actions = task_replays[step + 1].action()
                    rewards = task_replays[step + 1].reward()
                    dones = task_replays[step + 1].done()
                    next_states = task_replays[step + 1].next_state()
                    old_policy = old_policies[step + 1]
                    (old_density,
                     new_density,
                     old_log_probs,
                     new_log_probs) = precompute_quantities(states,
                                                            actions,
                                                            old_policy,
                                                            new_policy)

                    # Compute clip loss
                    advantages = compute_advantages(baseline, tau, gamma, rewards,
                                                    dones, states, next_states)
                    advantages = ch.normalize(advantages).detach()
                    clip_loss = ppo.policy_loss(new_log_probs,
                                                old_log_probs,
                                                advantages,
                                                clip=ppo_clip)

                    # Combine into ProMP loss
                    promp_loss += clip_loss + eta * kl_pen

            kl_total /= meta_bsz * adapt_steps
            promp_loss /= meta_bsz * adapt_steps
            opt.zero_grad()
            promp_loss.backward(retain_graph=True)
            opt.step()

            # Adapt KL penalty based on desired target
            if adaptive_penalty:
                if kl_total < kl_target / 1.5:
                    eta /= 2.0
                elif kl_total > kl_target * 1.5:
                    eta *= 2.0


if __name__ == '__main__':
    main()
