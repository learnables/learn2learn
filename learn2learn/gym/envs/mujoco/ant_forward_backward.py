#!/usr/bin/env python3

import gym
import numpy as np
from gym.envs.mujoco.mujoco_env import MujocoEnv

from learn2learn.gym.envs.meta_env import MetaEnv


class AntForwardBackwardEnv(MetaEnv, MujocoEnv, gym.utils.EzPickle):

    def __init__(self, task=None):
        MetaEnv.__init__(self, task)
        MujocoEnv.__init__(self, 'ant.xml', 5)
        gym.utils.EzPickle.__init__(self)

    # -------- MetaEnv Methods -------- 
    def set_task(self, task):
        MetaEnv.set_task(self, task)
        self.goal_direction = task['direction']

    def sample_tasks(self, num_tasks):
        directions = np.random.choice((-1, 1), (num_tasks, ))
        tasks = [{'direction': direction} for direction in directions]
        return tasks

    # -------- Mujoco Methods -------- 
    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        ])

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)

    # -------- Gym Methods -------- 
    def step(self, action):
        xposbefore = np.copy(self.get_body_com("torso")[0])
        self.do_simulation(action, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]
        forward_reward = self.goal_direction * (xposafter - xposbefore)/self.dt
        ctrl_cost = .5 * np.square(action).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() and 1.0 >= state[2] >= 0.
        done = not notdone
        ob = self._get_obs()
        return ob, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward)

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
    env = AntForwardBackwardEnv()
    for task in [env.get_task(), env.sample_tasks(1)[0]]:
        env.set_task(task)
        env.reset()
        action = env.action_space.sample()
        env.step(action)
