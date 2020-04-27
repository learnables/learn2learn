#!/usr/bin/env python3

from gym.error import DependencyNotInstalled


class MujocoEnv(object):
    """Dummy class to avoid Mujoco import errors"""

    def __init__(self, model_path=None, frame_skip=None):
        self.model_path = model_path
        self.frame_skip = frame_skip

    def _error(self):
        """docstring for _print_error"""
        msg = 'Importing MujocoEnv failed. Please run: \n' + \
              '\n`pip install mujoco-py` \n\n' + \
              'and visit: ' + \
              'https://github.com/openai/mujoco-py/'
        raise DependencyNotInstalled(msg)

    def __hasattr__(self, *args, **kwargs):
        """docstring for __hasattr__"""
        self._error()

    def __getattr__(self, *args, **kwargs):
        """docstring for __getattr__"""
        self._error()

    def reset(self, *args, **kwargs):
        """docstring for reset"""
        self._error()

    def step(self, *args, **kwargs):
        """docstring for reset"""
        self._error()

    def seed(self, *args, **kwargs):
        """docstring for reset"""
        self._error()
