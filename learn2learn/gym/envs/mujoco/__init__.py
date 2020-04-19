#!/usr/bin/env python3


class MujocoEnv(object):
    """Dummy class to avoid Mujoco import errors"""

    def __init__(self, model_path=None, frame_skip=None):
        self.model_path = model_path
        self.frame_skip = frame_skip

    def __hasattr__(self, *args, **kwargs):
        """docstring for __hasattr__"""
        raise 'Importing MujocoEnv failed. Please run: \n \
              `pip install mujoco-py` \n \
              and visit: \n \
              https://github.com/openai/mujoco-py/'


from .ant_direction import AntDirectionEnv
from .ant_forward_backward import AntForwardBackwardEnv
from .halfcheetah_forward_backward import HalfCheetahForwardBackwardEnv
from .humanoid_direction import HumanoidDirectionEnv
from .humanoid_forward_backward import HumanoidForwardBackwardEnv
