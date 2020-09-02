<p align="center"><img src="https://raw.githubusercontent.com/learnables/learn2learn/gh-pages/assets/img/l2l-full.png" height="120px" /></p>

--------------------------------------------------------------------------------

[![Build Status](https://travis-ci.com/learnables/learn2learn.svg?branch=master)](https://travis-ci.com/learnables/learn2learn)

learn2learn is a software library for meta-learning research.

learn2learn builds on top of PyTorch to accelerate two aspects of the meta-learning research cycle:

* *fast prototyping*, essential in letting researchers quickly try new ideas, and
* *correct reproducibility*, ensuring that these ideas are evaluated fairly.

learn2learn provides low-level utilities and unified interface to create new algorithms and domains, together with high-quality implementations of existing algorithms and standardized benchmarks.
It retains compatibility with [torchvision](https://pytorch.org/vision/), [torchaudio](https://pytorch.org/audio/), [torchtext](https://pytorch.org/text/), [cherry](http://cherry-rl.net/), and any other PyTorch-based library you might be using.

To learn more, see our whitepaper: [arXiv:2008.12284](https://arxiv.org/abs/2008.12284)

**Overview**

* [`learn2learn.data`](http://learn2learn.net/docs/learn2learn.data/): `TaskDataset` and transforms to create few-shot tasks from any PyTorch dataset.
* [`learn2learn.vision`](http://learn2learn.net/docs/learn2learn.vision/): Models, datasets, and benchmarks for computer vision and few-shot learning.
* [`learn2learn.gym`](http://learn2learn.net/docs/learn2learn.gym/): Environment and utilities for meta-reinforcement learning.
* [`learn2learn.algorithms`](http://learn2learn.net/docs/learn2learn.algorithms/): High-level wrappers for existing meta-learning algorithms.
* [`learn2learn.optim`](http://learn2learn.net/docs/learn2learn.optim/): Utilities and algorithms for differentiable optimization and meta-descent.

**Resources**

* Website: [http://learn2learn.net/](http://learn2learn.net/)
* Documentation: [http://learn2learn.net/docs/](http://learn2learn.net/docs/)
* Tutorials: [http://learn2learn.net/tutorials/getting_started/](http://learn2learn.net/tutorials/getting_started/)
* Examples: [https://github.com/learnables/learn2learn/tree/master/examples](https://github.com/learnables/learn2learn/tree/master/examples)
* GitHub: [https://github.com/learnables/learn2learn/](https://github.com/learnables/learn2learn/)
* Slack: [http://slack.learn2learn.net/](http://slack.learn2learn.net/)

## Installation

~~~bash
pip install learn2learn
~~~

## Snippets & Examples

The following snippets provide a sneak peek at the functionalities of learn2learn.

### High-level Wrappers

**Few-Shot Learning with MAML**

For more algorithms (ProtoNets, ANIL, Meta-SGD, Reptile, Meta-Curvature, KFO) refer to the [examples](https://github.com/learnables/learn2learn/tree/master/examples/vision) folder.
Most of them can be implemented with with the `GBML` wrapper. ([documentation](http://learn2learn.net/docs/learn2learn.algorithms/#gbml)).
~~~python
maml = l2l.algorithms.MAML(model, lr=0.1)
opt = torch.optim.SGD(maml.parameters(), lr=0.001)
for iteration in range(10):
    opt.zero_grad()
    task_model = maml.clone()  # torch.clone() for nn.Modules
    adaptation_loss = compute_loss(task_model)
    task_model.adapt(adaptation_loss)  # computes gradient, update task_model in-place
    evaluation_loss = compute_loss(task_model)
    evaluation_loss.backward()  # gradients w.r.t. maml.parameters()
    opt.step()
~~~

**Meta-Descent with Hypergradient**

Learn any kind of optimization algorithm with the `LearnableOptimizer`. ([example](https://github.com/learnables/learn2learn/tree/master/examples/optimization) and [documentation](http://learn2learn.net/docs/learn2learn.optim/#learnableoptimizer))
~~~python
linear = nn.Linear(784, 10)
transform = l2l.optim.ModuleTransform(l2l.nn.Scale)
metaopt = l2l.optim.LearnableOptimizer(linear, transform, lr=0.01)  # metaopt has .step()
opt = torch.optim.SGD(metaopt.parameters(), lr=0.001)  # metaopt also has .parameters()

metaopt.zero_grad()
opt.zero_grad()
error = loss(linear(X), y)
error.backward()
opt.step()  # update metaopt
metaopt.step()  # update linear
~~~

### Learning Domains

**Custom Few-Shot Dataset**

Many standardized datasets (Omniglot, mini-/tiered-ImageNet, FC100, CIFAR-FS) are readily available in `learn2learn.vision.datasets`.
([documentation](http://learn2learn.net/docs/learn2learn.vision/#learn2learnvisiondatasets))
~~~python
dataset = l2l.data.MetaDataset(MyDataset())  # any PyTorch dataset
transforms = [  # Easy to define your own transform
    l2l.data.transforms.NWays(dataset, n=5),
    l2l.data.transforms.KShots(dataset, k=1),
    l2l.data.transforms.LoadData(dataset),
]
taskset = TaskDataset(dataset, transforms, num_tasks=20000)
for task in taskset:
    X, y = task
    # Meta-train on the task
~~~

**Environments and Utilities for Meta-RL**

Parallelize your own meta-environments with `AsyncVectorEnv`, or use the standardized ones.
([documentation](http://learn2learn.net/docs/learn2learn.gym/#metaenv))
~~~python
def make_env():
    env = l2l.gym.HalfCheetahForwardBackwardEnv()
    env = cherry.envs.ActionSpaceScaler(env)
    return env

env = l2l.gym.AsyncVectorEnv([make_env for _ in range(16)])  # uses 16 threads
for task_config in env.sample_tasks(20):
    env.set_task(task)  # all threads receive the same task
    state = env.reset()  # use standard Gym API
    action = my_policy(env)
    env.step(action)
~~~

### Low-Level Utilities

**Differentiable Optimization**

Learn and differentiate through updates of PyTorch Modules.
([documentation](http://learn2learn.net/docs/learn2learn.optim/#parameterupdate))
~~~python

model = MyModel()
transform = l2l.optim.KroneckerTransform(l2l.nn.KroneckerLinear)
learned_update = l2l.optim.ParameterUpdate(  # learnable update function
        model.parameters(), transform)
clone = l2l.clone_module(model)  # torch.clone() for nn.Modules
error = loss(clone(X), y)
updates = learned_update(  # similar API as torch.autograd.grad
    error,
    clone.parameters(),
    create_graph=True,
)
l2l.update_module(clone, updates=updates)
loss(clone(X), y).backward()  # Gradients w.r.t model.parameters() and learned_update.parameters()
~~~

## Changelog

A human-readable changelog is available in the [CHANGELOG.md](CHANGELOG.md) file.

## Citation

To cite the `learn2learn` repository in your academic publications, please use the following reference.

> Arnold, Sebastien M. R., Praateek Mahajan, Debajyoti Datta, Ian Bunner, and Konstantinos Saitas Zarkias. 2020. “learn2learn: A Library for Meta-Learning Research.” arXiv [cs.LG]. http://arxiv.org/abs/2008.12284.

You can also use the following Bibtex entry.

~~~bib
@article{Arnold2020-ss,
  title         = "learn2learn: A Library for {Meta-Learning} Research",
  author        = "Arnold, S{\'e}bastien M R and Mahajan, Praateek and Datta,
                   Debajyoti and Bunner, Ian and Zarkias, Konstantinos Saitas",
  month         =  aug,
  year          =  2020,
  url           = "http://arxiv.org/abs/2008.12284",
  archivePrefix = "arXiv",
  primaryClass  = "cs.LG",
  eprint        = "2008.12284"
}

~~~

### Acknowledgements & Friends

1. The RL environments are adapted from Tristan Deleu's [implementations](https://github.com/tristandeleu/pytorch-maml-rl) and from the ProMP [repository](https://github.com/jonasrothfuss/ProMP/). Both shared with permission, under the MIT License.
2. [TorchMeta](https://github.com/tristandeleu/pytorch-meta) is similar library, with a focus on datasets for supervised meta-learning. 
3. [higher](https://github.com/facebookresearch/higher) is a PyTorch library that enables differentiating through optimization inner-loops. While they monkey-patch `nn.Module` to be stateless, learn2learn retains the stateful PyTorch look-and-feel. For more information, refer to [their ArXiv paper](https://arxiv.org/abs/1910.01727).
