#!/usr/bin/env python3

import gym
import numpy as np

from gym.error import DependencyNotInstalled
try:
    from gym.envs.mujoco.mujoco_env import MujocoEnv
except DependencyNotInstalled:
    from learn2learn.gym.envs.mujoco.dummy_mujoco_env import MujocoEnv

from learn2learn.gym.envs.meta_env import MetaEnv


class HalfCheetahForwardBackwardEnv(MetaEnv, MujocoEnv, gym.utils.EzPickle):
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/gym/envs/mujoco/halfcheetah_forward_backward.py)

    **Description**

    This environment requires the half-cheetah to learn to run forward or backward.
    At each time step the half-cheetah receives a signal composed of a
    control cost and a reward equal to its average velocity in the direction
    of the plane. The tasks are Bernoulli samples on {-1, 1} with probability 0.5, where -1 indicates the half-cheetah should
    move backward and +1 indicates the half-cheetah should move forward.
    The velocity is calculated as the distance (in the target direction) of the half-cheetah's torso
    position before and after taking the specified action divided by a small value dt.

    **Credit**

    Adapted from Jonas Rothfuss' implementation.

    **References**

    1. Finn et al. 2017. "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks." arXiv [cs.LG].
    2. Rothfuss et al. 2018. "ProMP: Proximal Meta-Policy Search." arXiv [cs.LG].

    """

    def __init__(self, task=None):
        MetaEnv.__init__(self, task)
        MujocoEnv.__init__(self, 'half_cheetah.xml', 5)
        gym.utils.EzPickle.__init__(self)

    # -------- MetaEnv Methods --------
    def set_task(self, task):
        MetaEnv.set_task(self, task)
        self.goal_direction = task['direction']

    def sample_tasks(self, num_tasks):
        directions = np.random.choice((-1.0, 1.0), (num_tasks,))
        tasks = [{'direction': direction} for direction in directions]
        return tasks

    # -------- Mujoco Methods --------
    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat,
            self.get_body_com("torso").flat,
        ]).astype(np.float32).flatten()

    def viewer_setup(self):
        camera_id = self.model.camera_name2id('track')
        self.viewer.cam.type = 2
        self.viewer.cam.fixedcamid = camera_id
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        # Hide the overlay
        self.viewer._hide_overlay = True

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    # -------- Gym Methods --------
    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]

        forward_vel = (xposafter - xposbefore) / self.dt
        forward_reward = self.goal_direction * forward_vel
        ctrl_cost = 0.5 * 1e-1 * np.sum(np.square(action))

        observation = self._get_obs()
        reward = forward_reward - ctrl_cost
        done = False
        infos = dict(reward_forward=forward_reward,
                     reward_ctrl=-ctrl_cost, task=self._task)
        return (observation, reward, done, infos)

    def reset(self, *args, **kwargs):
        MujocoEnv.reset(self, *args, **kwargs)
        return self._get_obs()

    def render(self, mode='human'):
        if mode == 'rgb_array':
            self._get_viewer(mode).render()
            # window size used for old mujoco-py:
            width, height = 500, 500
            data = self._get_viewer(mode).read_pixels(width,
                                                      height,
                                                      depth=False)
            return data
        elif mode == 'human':
            self._get_viewer(mode).render()


if __name__ == '__main__':
    env = HalfCheetahForwardBackwardEnv()
    for task in [env.get_task(), env.sample_tasks(1)[0]]:
        env.set_task(task)
        env.reset()
        action = env.action_space.sample()
        env.step(action)
