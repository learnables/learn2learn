#!/usr/bin/env python3

import gym
import numpy as np

from gym.error import DependencyNotInstalled
try:
    from gym.envs.mujoco.mujoco_env import MujocoEnv
except DependencyNotInstalled:
    from learn2learn.gym.envs.mujoco.dummy_mujoco_env import MujocoEnv

from learn2learn.gym.envs.meta_env import MetaEnv


def mass_center(model, sim):
    mass = np.expand_dims(model.body_mass, 1)
    xpos = sim.data.xipos
    return (np.sum(mass * xpos, 0) / np.sum(mass))


class HumanoidDirectionEnv(MetaEnv, MujocoEnv, gym.utils.EzPickle):
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/gym/envs/mujoco/humanoid_direction.py)

    **Description**

    This environment requires the humanoid to learn to run in a random direction in the
    XY plane. At each time step the humanoid receives a signal composed of a
    control cost and a reward equal to its average velocity in the target direction.
    The tasks are 2d-arrays sampled uniformly along the unit circle.
    The target direction is indicated by the vector from the origin to the sampled point.
    The velocity is calculated as the distance (in the target direction) of the humanoid's torso
    position before and after taking the specified action divided by a small value dt.
    A small positive bonus is added to the reward to stop the humanoid from
    prematurely ending the episode.

    **Credit**

    Adapted from Jonas Rothfuss' implementation.

    **References**

    1. Finn et al. 2017. "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks." arXiv [cs.LG].
    2. Rothfuss et al. 2018. "ProMP: Proximal Meta-Policy Search." arXiv [cs.LG].

    """

    def __init__(self, task=None):
        MetaEnv.__init__(self, task)
        MujocoEnv.__init__(self, 'humanoid.xml', 5)
        gym.utils.EzPickle.__init__(self)

    # -------- MetaEnv Methods --------
    def set_task(self, task):
        MetaEnv.set_task(self, task)
        self.goal_direction = task['direction']

    def sample_tasks(self, num_tasks):
        directions = np.random.normal(size=(num_tasks, 2))
        directions /= np.linalg.norm(directions, axis=1)[..., np.newaxis]
        tasks = [{'direction': direction} for direction in directions]
        return tasks

    # -------- Mujoco Methods --------
    def _get_obs(self):
        data = self.sim.data
        return np.concatenate([data.qpos.flat[2:],
                               data.qvel.flat,
                               data.cinert.flat,
                               data.cvel.flat,
                               data.qfrc_actuator.flat,
                               data.cfrc_ext.flat])

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 1.0
        self.viewer.cam.elevation = -20

    def reset_model(self):
        c = 0.01
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-c, high=c, size=self.model.nv, )
        )
        return self._get_obs()

    # -------- Gym Methods --------
    def step(self, action):
        pos_before = np.copy(mass_center(self.model, self.sim)[:2])
        self.do_simulation(action, self.frame_skip)
        pos_after = mass_center(self.model, self.sim)[:2]
        alive_bonus = 5.0
        data = self.sim.data
        lin_vel_cost = 0.25 * np.sum(self.goal_direction * (pos_after - pos_before)) / self.model.opt.timestep
        quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum()
        quad_impact_cost = .5e-6 * np.square(data.cfrc_ext).sum()
        quad_impact_cost = min(quad_impact_cost, 10)
        reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus
        qpos = self.sim.data.qpos
        done = bool((qpos[2] < 1.0) or (qpos[2] > 2.0))
        return self._get_obs(), reward, done, dict(reward_linvel=lin_vel_cost,
                                                   reward_quadctrl=-quad_ctrl_cost,
                                                   reward_alive=alive_bonus,
                                                   reward_impact=-quad_impact_cost)

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
    env = HumanoidDirectionEnv()
    for task in [env.get_task(), env.sample_tasks(1)[0]]:
        env.set_task(task)
        env.reset()
        action = env.action_space.sample()
        env.step(action)
