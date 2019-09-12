#!/usr/bin/env python3

import numpy as np
from gym import spaces
from gym.utils import seeding

from learn2learn.gym.envs.meta_env import MetaEnv


class Particles2DEnv(MetaEnv):
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/gym/envs/particles/particles_2d.py)

    **Description**

    Each task is defined by the location of the goal. A point mass
    receives a directional force and moves accordingly
    (clipped in [-0.1,0.1]). The reward is equal to the negative
    distance from the goal.

    **Credit**

    Adapted from Jonas Rothfuss' implementation.

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
        """
        Tasks correspond to a goal point chosen uniformly at random.
        """
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
        """
        Sets point mass position back to (0,0)
        """
        self._state = np.zeros(2, dtype=np.float32)
        return self._state

    def step(self, action):
        """
        **Description**

        Given an action, clips the action to be in the
        appropriate range and moves the point mass position
        according to the action.

        **Arguments**

        action (2-element array) - Array specifying the magnitude
        and direction of the forces to be applied in the x and y
        planes.

        **Returns**

        *state, reward, done, task*

        * state (arr) - is a 2-element array encoding the x,y position of
        the point mass

        * reward (float) - signal equal to the negative squared distance
        from the goal

        * done (bool) - boolean indicating whether or not the point mass
        is epsilon or less distance from the goal

        * task (dict) - dictionary of task specific parameters and their current
        values

        """
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
