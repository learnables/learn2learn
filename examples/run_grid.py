import gym
from gym.wrappers import Monitor
import cherry as ch
import learn2learn as l2l
import torch
import random

RENDER = True

ENV_SIZE = 11
START = (int((ENV_SIZE-1)/2), int((ENV_SIZE-1)/2))
GOAL = (1,1)

def get_random_action(state):
    action = torch.tensor([[random.randint(0,3)]])
    return action

def main():

    env = 'MiniGrid-Empty-v0'
    env = gym.make(env, size=ENV_SIZE)
    env = ch.envs.Torch(env)
    env = ch.envs.Runner(env)
    env = Monitor(env, "./vid", video_callable=lambda episode_id: True,force=True)

    for task_config in env.sample_tasks(4):
        env.reset_task(task_config)
        env.reset()
        transition = env.run(get_random_action, episodes=5, render=RENDER)

if __name__ == '__main__':
    main()
