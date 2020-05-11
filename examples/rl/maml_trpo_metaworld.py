#!/usr/bin/env python3

"""
Trains a 2-layer MLP with MAML-TRPO on the 'ML' benchmarks of metaworld.
For more information related to the benchmark check out https://github.com/rlworkgroup/metaworld

Usage:

python examples/rl/maml_trpo.py
"""

import random
from copy import deepcopy

import metaworld.benchmarks as mtwrld

import cherry as ch
import numpy as np
import torch
from cherry.algorithms import a2c, trpo
from cherry.models.robotics import LinearValue
from torch import autograd
from torch.distributions.kl import kl_divergence
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from tqdm import tqdm

import learn2learn as l2l

from learn2learn.gym.envs.metaworld import MetaWorldML1 as ML1
from learn2learn.gym.envs.metaworld import MetaWorldML10 as ML10
from learn2learn.gym.envs.metaworld import MetaWorldML45 as ML45
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
    second_order = not first_order
    loss = maml_a2c_loss(train_episodes, clone, baseline, gamma, tau)
    gradients = autograd.grad(loss,
                              clone.parameters(),
                              retain_graph=second_order,
                              create_graph=second_order)
    return l2l.algorithms.maml.maml_update(clone, adapt_lr, gradients)


def meta_surrogate_loss(iteration_replays, iteration_policies, policy, baseline, tau, gamma, adapt_lr):
    mean_loss = 0.0
    mean_kl = 0.0
    for task_replays, old_policy in tqdm(zip(iteration_replays, iteration_policies),
                                         total=len(iteration_replays),
                                         desc='Surrogate Loss',
                                         leave=False):
        train_replays = task_replays[:-1]
        valid_episodes = task_replays[-1]
        new_policy = l2l.clone_module(policy)

        # Fast Adapt
        for train_episodes in train_replays:
            new_policy = fast_adapt_a2c(new_policy, train_episodes, adapt_lr,
                                        baseline, gamma, tau, first_order=False)

        # Useful values
        states = valid_episodes.state()
        actions = valid_episodes.action()
        next_states = valid_episodes.next_state()
        rewards = valid_episodes.reward()
        dones = valid_episodes.done()

        # Compute KL
        old_densities = old_policy.density(states)
        new_densities = new_policy.density(states)
        kl = kl_divergence(new_densities, old_densities).mean()
        mean_kl += kl

        # Compute Surrogate Loss
        advantages = compute_advantages(baseline, tau, gamma, rewards, dones, states, next_states)
        advantages = ch.normalize(advantages).detach()
        old_log_probs = old_densities.log_prob(actions).mean(dim=1, keepdim=True).detach()
        new_log_probs = new_densities.log_prob(actions).mean(dim=1, keepdim=True)
        mean_loss += trpo.policy_loss(new_log_probs, old_log_probs, advantages)
    mean_kl /= len(iteration_replays)
    mean_loss /= len(iteration_replays)
    return mean_loss, mean_kl


def make_env(benchmark, seed, num_workers, test=False):
    # Set a specific task or left empty to train on all available tasks
    task = 'pick-place-v1' if benchmark == ML1 else False  # In this case, False corresponds to the sample_all argument

    def init_env():
        if test:
            env = benchmark.get_test_tasks(task)
        else:
            env = benchmark.get_train_tasks(task)

        env = ch.envs.ActionSpaceScaler(env)
        return env

    env = l2l.gym.AsyncVectorEnv([init_env for _ in range(num_workers)])

    env.seed(seed)
    env.set_task(env.sample_tasks(1)[0])
    env = ch.envs.Torch(env)
    return env


def main(
        benchmark=ML10,  # Choose between ML1, ML10, ML45
        adapt_lr=0.1,
        meta_lr=0.05,
        adapt_steps=3,
        num_iterations=10000,
        meta_bsz=10,
        adapt_bsz=10,  # Number of episodes to sample PER WORKER!
        tau=1.00,
        gamma=0.99,
        seed=42,
        num_workers=1,
        cuda=0):

    env = make_env(benchmark, seed, num_workers)

    cuda = bool(cuda)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

    policy = DiagNormalPolicy(env.state_size, env.action_size)
    if cuda:
        policy.to('cuda')
    baseline = LinearValue(env.state_size, env.action_size)

    for iteration in range(num_iterations):
        iteration_reward = 0.0
        iteration_replays = []
        iteration_policies = []

        for task_config in tqdm(env.sample_tasks(meta_bsz), leave=False, desc='Data'):  # Samples a new config
            clone = deepcopy(policy)
            env.set_task(task_config)
            env.reset()
            task = ch.envs.Runner(env)
            task_replay = []

            # Fast Adapt
            for step in range(adapt_steps):
                train_episodes = task.run(clone, episodes=adapt_bsz)
                clone = fast_adapt_a2c(clone, train_episodes, adapt_lr, baseline, gamma, tau, first_order=True)
                task_replay.append(train_episodes)

            # Compute Validation Loss
            valid_episodes = task.run(clone, episodes=adapt_bsz)
            task_replay.append(valid_episodes)

            iteration_reward += valid_episodes.reward().sum().item() / adapt_bsz
            iteration_replays.append(task_replay)
            iteration_policies.append(clone)

        # Print statistics
        print('\nIteration', iteration)
        validation_reward = iteration_reward / meta_bsz
        print('validation_reward', validation_reward)

        # TRPO meta-optimization
        backtrack_factor = 0.5
        ls_max_steps = 15
        max_kl = 0.01
        if cuda:
            policy.to('cuda', non_blocking=True)
            baseline.to('cuda', non_blocking=True)
            iteration_replays = [[r.to('cuda', non_blocking=True) for r in task_replays] for task_replays in
                                 iteration_replays]

        # Compute CG step direction
        old_loss, old_kl = meta_surrogate_loss(iteration_replays, iteration_policies, policy, baseline, tau, gamma,
                                               adapt_lr)
        grad = autograd.grad(old_loss,
                             policy.parameters(),
                             retain_graph=True)
        grad = parameters_to_vector([g.detach() for g in grad])
        Fvp = trpo.hessian_vector_product(old_kl, policy.parameters())
        step = trpo.conjugate_gradient(Fvp, grad)
        shs = 0.5 * torch.dot(step, Fvp(step))
        lagrange_multiplier = torch.sqrt(shs / max_kl)
        step = step / lagrange_multiplier
        step_ = [torch.zeros_like(p.data) for p in policy.parameters()]
        vector_to_parameters(step, step_)
        step = step_
        del old_kl, Fvp, grad
        old_loss.detach_()

        # Line-search
        for ls_step in range(ls_max_steps):
            stepsize = backtrack_factor ** ls_step * meta_lr
            clone = deepcopy(policy)
            for p, u in zip(clone.parameters(), step):
                p.data.add_(-stepsize, u.data)
            new_loss, kl = meta_surrogate_loss(iteration_replays, iteration_policies, clone, baseline, tau, gamma,
                                               adapt_lr)
            if new_loss < old_loss and kl < max_kl:
                for p, u in zip(policy.parameters(), step):
                    p.data.add_(-stepsize, u.data)
                break

    # Evaluate on a set of unseen tasks
    evaluate(benchmark, policy, baseline, adapt_lr, gamma, tau, num_workers, seed)


def evaluate(benchmark, policy, baseline, adapt_lr, gamma, tau, n_workers, seed):
    # Parameters
    adapt_steps = 3
    adapt_bsz = 4  # PER WORKER
    n_eval_tasks = 20

    tasks_reward = 0

    env = make_env(benchmark, seed, n_workers, test=True)
    eval_task_list = env.sample_tasks(n_eval_tasks)

    for i, task in enumerate(eval_task_list):
        clone = deepcopy(policy)
        env.set_task(task)
        env.reset()
        task = ch.envs.Runner(env)

        # Adapt
        for step in range(adapt_steps):
            adapt_episodes = task.run(clone, episodes=adapt_bsz)
            clone = fast_adapt_a2c(clone, adapt_episodes, adapt_lr, baseline, gamma, tau, first_order=True)
            task.env.reset()

        eval_episodes = task.run(clone, episodes=adapt_bsz)

        task_reward = eval_episodes.reward().sum().item() / adapt_bsz
        print(f"Reward for task {i} : {task_reward}")
        tasks_reward += task_reward

    final_eval_reward = tasks_reward / n_eval_tasks

    print(f"Average reward over {n_eval_tasks} test tasks: {final_eval_reward}")

    return final_eval_reward


if __name__ == '__main__':
    main()
