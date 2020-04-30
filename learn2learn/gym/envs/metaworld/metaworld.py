#!/usr/bin/env python3

import numpy as np
from gym import spaces
from gym.utils import seeding

from learn2learn.gym.envs.meta_env import MetaEnv


class MetaWorld(MetaEnv):
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/gym/envs/metaworld/metaworld.py)

    **Description**


    **Credit**



    """

    def __init__(self, task=None):
        self.seed()
        super(MetaWorld, self).__init__(task)
        self.observation_space = spaces.Box()
        self.action_space = spaces.Box()
        self.reset()

    # -------- MetaEnv Methods --------
    def sample_tasks(self, num_tasks):
        """
        """
        raise NotImplementedError

    def set_task(self, task):
        self._task = task
        self._goal = task['goal']

    # -------- Gym Methods --------
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, env=True):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

    def render(self, mode=None):
        raise NotImplementedError
