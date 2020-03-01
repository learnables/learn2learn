
# Meta-Reinforcement Learning

!!! warning
    Meta-RL results are particularly finicky to compare.
    Different papers use different environment implementations, which in turn produce different convergence and rewards.
    The plots below only serve to indicate what kind of performance you can expect with learn2learn.

## MAML

The following results are obtained by running [maml_trpo.py](https://github.com/learnables/learn2learn/blob/master/examples/rl/maml_trpo.py) on `HalfCheetahForwardBackwardEnv` and `AntForwardBackwardEnv` for 300 updates.
The figures show the expected sum of rewards over all tasks.
The line and shadow are the mean and standard deviation computed over 3 random seeds.

![](assets/img/examples/cheetah_fwdbwd_rewards.png) &
![](assets/img/examples/ant_fwdbwd_rewards.png)
