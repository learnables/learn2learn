from gym.envs.registration import load
from .normalized_env import NormalizedActionWrapper

def mujoco_wrapper(entry_point, **kwargs):
    # Load the environment from its entry point
    env_cls = load(entry_point)
    env = env_cls(**kwargs)
    # Normalization wrapper
    env = NormalizedActionWrapper(env)
    return env
