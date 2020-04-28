
# Meta-Reinforcement Learning

!!! warning
    Meta-RL results are particularly finicky to compare.
    Different papers use different environment implementations, which in turn produce different convergence and rewards.
    The plots below only serve to indicate what kind of performance you can expect with learn2learn.

## MAML

<p align="center">
<img src="http://learn2learn.net/assets/img/examples/cheetah_fwdbwd_rewards.png" height="330px" />
<img src="http://learn2learn.net/assets/img/examples/ant_fwdbwd_rewards.png" height="330px" />
</p>

The above results are obtained by running [maml_trpo.py](https://github.com/learnables/learn2learn/blob/master/examples/rl/maml_trpo.py) on `HalfCheetahForwardBackwardEnv` and `AntForwardBackwardEnv` for 300 updates.
The figures show the expected sum of rewards over all tasks.
The line and shadow are the mean and standard deviation computed over 3 random seeds.

### [Meta-World](https://github.com/rlworkgroup/metaworld)

The file [maml_trpo_metaworld.py](https://github.com/learnables/learn2learn/blob/master/examples/rl/maml_trpo_metaworld.py) runs MAML-TRPO on the meta-learning benchmarks ML1, ML10 and ML45. Training on ML1 is relatively fast and stable, the ML10 and ML45 benchmarks however are much more difficult and might be sensitive to hyper-parameter changes.

!!! info
    Those results were obtained in August 2019, and might be outdated.
