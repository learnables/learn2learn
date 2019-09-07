from gym.envs.registration import register

from .subproc_vec_env import SubprocVecEnv

# Bandit
# ----------------------------------------

for k in [5, 10, 50]:
    register(
        'Bandit-K{0}-v0'.format(k),
        entry_point='learn2learn.gym.envs.bandit:BernoulliBanditEnv',
        kwargs={'k': k}
    )

# TabularMDP
# ----------------------------------------

register(
    'TabularMDP-v0',
    entry_point='learn2learn.gym.envs.mdp:TabularMDPEnv',
    kwargs={'num_states': 10, 'num_actions': 5},
    max_episode_steps=10
)

# Mujoco
# ----------------------------------------

register(
    'HalfCheetahForwardBackward-v1',
    entry_point='learn2learn.gym.envs.mujoco.halfcheetah_forward_backward:HalfCheetahForwardBackwardEnv',
    max_episode_steps=100,
)

register(
    'AntForwardBackward-v1',
    entry_point='learn2learn.gym.envs.mujoco.ant_forward_backward:AntForwardBackwardEnv',
    max_episode_steps=100,
)

register(
    'AntDirection-v1',
    entry_point='learn2learn.gym.envs.mujoco.ant_direction:AntDirectionEnv',
    max_episode_steps=100,
)

register(
    'HumanoidForwardBackward-v1',
    entry_point='learn2learn.gym.envs.mujoco.humanoid_forward_backward:HumanoidForwardBackwardEnv',
    max_episode_steps=200,
)

register(
    'HumanoidDirection-v1',
    entry_point='learn2learn.gym.envs.mujoco.humanoid_direction:HumanoidDirectionEnv',
    max_episode_steps=200,
)






# 2D Navigation
# ----------------------------------------

register(
    '2DNavigation-v0',
    entry_point='learn2learn.gym.envs.navigation:Navigation2DEnv',
    max_episode_steps=100
)
