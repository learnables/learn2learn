#!/usr/bin/env python3

from gym.core import Env


class MetaEnv(Env):
    """
    """

    def __init__(self, task=None):
        super(Env, self).__init__()
        if task is None:
            task = self.sample_tasks(1)[0]
        self.set_task(task)

    def sample_tasks(self, num_tasks):
        """
        """
        raise NotImplementedError

    def set_task(self, task):
        """
        """
        self._task = task

    def get_task(self):
        """
        """
        return self._task
