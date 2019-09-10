#!/usr/bin/env python3

import numpy as np
from gym import spaces
from gym.utils import seeding

from learn2learn.gym.envs.meta_env import MetaEnv


class Particles2DEnv(MetaEnv):
    """
    """

    def __init__(self, task=None):
        self.seed()
        super(Particles2DEnv, self).__init__(task)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(2,), dtype=np.float32)
        self.action_space = spaces.Box(low=-0.1, high=0.1,
                                       shape=(2,), dtype=np.float32)
        self.reset()

    # -------- MetaEnv Methods -------- 
    def sample_tasks(self, num_tasks):
        goals = self.np_random.uniform(-0.5, 0.5, size=(num_tasks, 2))
        tasks = [{'goal': goal} for goal in goals]
        return tasks

    def set_task(self, task):
        self._task = task
        self._goal = task['goal']

    # -------- Gym Methods -------- 
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, env=True):
        self._state = np.zeros(2, dtype=np.float32)
        return self._state

    def step(self, action):
        action = np.clip(action, -0.1, 0.1)
        assert self.action_space.contains(action)
        self._state = self._state + action

        x = self._state[0] - self._goal[0]
        y = self._state[1] - self._goal[1]
        reward = -np.sqrt(x ** 2 + y ** 2)
        done = ((np.abs(x) < 0.01) and (np.abs(y) < 0.01))

        return self._state, reward, done, self._task

    def render(self, mode=None):
        raise NotImplementedError
