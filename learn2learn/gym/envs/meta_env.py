#!/usr/bin/env python3

from gym.core import Env


class MetaEnv(Env):
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/gym/envs/meta_env.py)

    **Description**

    Interface for l2l envs.

    **Credit**

    Inspired by tristandeleu's implementation of MAML paper environments.

    """

    def __init__(self, task=None):
        super(Env, self).__init__()
        if task is None:
            task = self.sample_tasks(1)[0]
        self.set_task(task)

    def sample_tasks(self, num_tasks):
        """
        **Description**

        Samples num_tasks tasks for training or evaluation.
        How the tasks are sampled determines the task distribution.

        **Arguments**

        num_tasks (int) - number of tasks to sample

        **Returns**

        tasks ([dict]) - returns a list of num_tasks tasks. Tasks are 
        dictionaries of task specific parameters. A
        minimal example for num_tasks = 1 is [{'goal': value}].
        """
        raise NotImplementedError

    def set_task(self, task):
        """
        **Description**

        Sets the task specific parameters defined in task. 

        **Arguments**

        task (dict) - A dictionary of task specific parameters and
        their values.

        **Returns**

        None.
        """
        self._task = task

    def get_task(self):
        """
        **Description**

        Returns the current task.

        **Arguments**

        None.

        **Returns**

        (task) - Dictionary of task specific parameters and their
        current values.
        """
        return self._task
