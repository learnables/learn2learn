#!/usr/bin/env python3

from gym.error import DependencyNotInstalled
# This exception is not enough. Maybe, we would need dummies for the ML benchmarks as well.
try:
    from metaworld.envs.mujoco.multitask_env import MultiClassMultiTaskEnv
    from metaworld.benchmarks import ML1, ML10, ML45
except (DependencyNotInstalled, ModuleNotFoundError):
    from learn2learn.gym.envs.mujoco.dummy_mujoco_env import MujocoEnv as MultiClassMultiTaskEnv

    class ML1:
        pass

    class ML10:
        pass

    class ML45:
        pass

from learn2learn.gym.envs.meta_env import MetaEnv


class MetaWorldMod(MultiClassMultiTaskEnv, MetaEnv):
    """
    This wrapper inherits from the same base class MultiClassMultiTaskEnv that the ML1,
    ML10, ML45 benchmarks use and also L2L's MetaEnv wrapper. The reason for this wrapper
    is to re-enable the functionality of the done signal (return true) when the agent reaches
    the end of the horizon (timestep limit) on the environment. The developers of MetaWorld
    decided to disabled it due to some issues that was causing in off-policy algorithms.
    For more information, please see PR https://github.com/rlworkgroup/metaworld/pull/45
    """
    def __init__(self, task_env_cls_dict, task_args_kwargs, sample_all=True, sample_goals=False, obs_type='plain'):
        super(MetaWorldMod, self).__init__(task_env_cls_dict=task_env_cls_dict,
                                           task_args_kwargs=task_args_kwargs,
                                           sample_goals=sample_goals,
                                           obs_type=obs_type,
                                           sample_all=sample_all)
        self.collected_steps = 0

    # -------- Gym Methods --------
    def step(self, action):
        obs, reward, done, info = super().step(action)
        self.collected_steps += 1

        # Manually set done at the end of the horizon
        if self.collected_steps >= self.active_env.max_path_length:
            done = True

        return obs, reward, done, info

    def reset(self, **kwargs):
        self.collected_steps = 0
        obs = super().reset(**kwargs)
        return obs

    # -------- MetaEnv Methods --------
    def sample_tasks(self, meta_batch_size):
        return super().sample_tasks(meta_batch_size)

    def set_task(self, task):
        return super().set_task(task)

    def get_task(self):
        return super().get_task()


class MetaWorldML1(ML1, MetaWorldMod):
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/gym/envs/metaworld/metaworld.py)

    **Description**

    The ML1 Benchmark of Meta-World is focused on solving just one task on different object / goal
    configurations.This task can be either one of the following: 'reach', 'push' and 'pick-and-place'.
    The meta-training is performed on a set of 50 randomly chosen once initial object and goal positions.
    The meta-testing is performed on a held-out set of 10 new different configurations. The starting state
    of the robot arm is always fixed. The goal positions are not provided in the observation space, forcing
    the Sawyer robot arm to explore and adapt to the new goal through trial-and-error. This is considered
    a relatively easy problem for a meta-learning algorithm to solve and acts as a sanity check to a
    working implementation. For more information regarding this benchmark, please consult [1].

    **Credit**

    Original implementation found in https://github.com/rlworkgroup/metaworld.

    **References**

    1. Yu, Tianhe, et al. "Meta-world: A benchmark and evaluation for multi-task and meta reinforcement learning."
    arXiv preprint arXiv:1910.10897 (2019).

    """

    def __init__(self, task_name, env_type='train', n_goals=50, sample_all=False):
        super(MetaWorldML1, self).__init__(task_name, env_type, n_goals, sample_all)


class MetaWorldML10(ML10, MetaWorldMod):
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/gym/envs/metaworld/metaworld.py)

    **Description**

    The ML10 Benchmark of Meta-World consists of 10 different tasks for meta-training and 5 new tasks for
    meta-testing. For each task there is only one goal that is randomly chosen once. The starting state and
    object position is random. The meta-training tasks have been intentionally selected to have a
    structural similarity to the test tasks. No task ID is provided in the observation space, meaning the
    meta-learning algorithm will need to identify each task from experience. This is a much harder problem
    than ML1 which probably requires more samples to train. For more information regarding this benchmark,
    please consult [1].

    **Credit**

    Original implementation found in https://github.com/rlworkgroup/metaworld.

    **References**

    1. Yu, Tianhe, et al. "Meta-world: A benchmark and evaluation for multi-task and meta reinforcement learning."
    arXiv preprint arXiv:1910.10897 (2019).

    """

    def __init__(self, env_type='train', sample_all=False, task_name=None):
        super(MetaWorldML10, self).__init__(env_type, sample_all, task_name)


class MetaWorldML45(ML45, MetaWorldMod):
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/gym/envs/metaworld/metaworld.py)

    **Description**

    Similarly to ML10, this Benchmark has a variety of 45 different tasks for meta-training and 5 new tasks for
    meta-testing. For each task there is only one goal that is randomly chosen once. The starting state and
    object position is random. No task ID is provided in the observation space, meaning the meta-learning
    algorithm will need to identify each task from experience. This benchmark is significantly difficult to
    solve due to the diversity across tasks. For more information regarding this benchmark, please consult [1].

    **Credit**

    Original implementation found in https://github.com/rlworkgroup/metaworld.

    **References**

    1. Yu, Tianhe, et al. "Meta-world: A benchmark and evaluation for multi-task and meta reinforcement learning."
    arXiv preprint arXiv:1910.10897 (2019).

    """

    def __init__(self, env_type='train', sample_all=False, task_name=None):
        super(MetaWorldML45, self).__init__(env_type, sample_all, task_name)
