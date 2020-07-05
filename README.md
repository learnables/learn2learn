<p align="center"><img src="https://raw.githubusercontent.com/learnables/learn2learn/gh-pages/assets/img/l2l-full.png" height="120px" /></p>

--------------------------------------------------------------------------------

[![Build Status](https://travis-ci.com/learnables/learn2learn.svg?branch=master)](https://travis-ci.com/learnables/learn2learn)

learn2learn is a software library for meta-learning research.

learn2learn builds on top of PyTorch to accelerate two aspects of the meta-learning research cycle:

* *fast prototyping*, essential in letting researchers quickly try new ideas, and
* *correct reproducibility*, ensuring that these ideas are evaluated fairly

learn2learn provides low-level utilities and unified interface to create new algorithms and domains, together with high-quality implementations of existing algorithms and standardized benchmarks.

**Overview**

* [`learn2learn.data`](http://learn2learn.net/docs/learn2learn.data/): `TaskDataset` and its transforms, to quickly create few-shot learning datasets.
* [`learn2learn.vision`](http://learn2learn.net/docs/learn2learn.vision/): Models, datasets, and benchmarks for few-shot learning in computer vision.
* [`learn2learn.gym`](http://learn2learn.net/docs/learn2learn.gym/): Environment and utilities for meta-reinforcement learning.
* [`learn2learn.algorithms`](http://learn2learn.net/docs/learn2learn.algorithms/): High-level wrappers for existing meta-learning methods.
* [`learn2learn.optim`](http://learn2learn.net/docs/learn2learn.optim/): Utilities and algorithms for differentiable optimization and meta-descent.

**Resources**

* Website: [http://learn2learn.net/](http://learn2learn.net/)
* Documentation: [http://learn2learn.net/docs/](http://learn2learn.net/docs/)
* Examples: [https://github.com/learnables/learn2learn/tree/master/examples](https://github.com/learnables/learn2learn/tree/master/examples)
* GitHub: [https://github.com/learnables/learn2learn/](https://github.com/learnables/learn2learn/)
* Slack: [http://slack.learn2learn.net/](http://slack.learn2learn.net/)


## Installation

~~~bash
pip install learn2learn
~~~

## Snippets & Examples

For each snippet, point to full-fledged examples and documentation.

### High-level Wrappers

~~~python
maml = l2l.algorithms.MAML(model, lr=0.1)
opt = torch.optim.SGD(gbml.parameters(), lr=0.001)
for iteration in range(10):
    opt.zero_grad()
    task_model = maml.clone()
    adaptation_loss = compute_loss(task_model)
    task_model.adapt(adaptation_loss)  # computes gradient, update task_model in-place
    evaluation_loss = compute_loss(task_model)
    evaluation_loss.backward()
    opt.step()
~~~

~~~python
linear = nn.Linear(784, 10)
transform = l2l.optim.ModuleTransform(torch.nn.Linear)
metaopt = l2l.optim.LearnableOptimizer(linear, transform, lr=0.01)
opt = torch.optim.SGD(metaopt.parameters(), lr=0.001)

metaopt.zero_grad()
opt.zero_grad()
error = loss(linear(X), y)
error.backward()
opt.step()  # update metaopt
metaopt.step()  # update linear
~~~

### Low-Level Utilities

~~~python
error = loss(model(X), y)
grads = torch.autograd.grad(
    error,
    model.parameters(),
    create_graph=True,
)
updates = [-lr * g for g in grads]
l2l.update_module(model, updates=updates)
~~~

### Few-Shot and Reinforcement Learning Domains

~~~python
dataset = l2l.data.MetaDataset(MyDataset())
transforms = [
    l2l.data.transforms.NWays(dataset, n=5),
    l2l.data.transforms.KShots(dataset, k=1),
    l2l.data.transforms.LoadData(dataset),
]
taskset = TaskDataset(dataset, transforms, num_tasks=20000)
for task in taskset:
    X, y = task
~~~

~~~python
def make_env():
    env = l2l.gym.HalfCheetahForwardBackwardEnv()
    env = cherry.envs.ActionSpaceScaler(env)
    return env

env = l2l.gym.AsyncVectorEnv([make_env for _ in range(16)])
for task_config in env.sample_tasks(20):
    env.set_task(task)
    action = my_policy(env)
    env.step(action)
~~~


## Documentation

Documentation and tutorials are available on learn2learn’s website: [http://learn2learn.net](http://learn2learn.net).

## Changelog

A human-readable changelog is available in the [CHANGELOG.md](CHANGELOG.md) file.

## Citation

To cite the `learn2learn` repository in your academic publications, please use the following reference.

> Sebastien M.R. Arnold, Praateek Mahajan, Debajyoti Datta, Ian Bunner. `"learn2learn"`. [https://github.com/learnables/learn2learn](https://github.com/learnables/learn2learn), 2019.

You can also use the following Bibtex entry.

~~~bib
@misc{learn2learn2019,
    author       = {Sebastien M.R. Arnold, Praateek Mahajan, Debajyoti Datta, Ian Bunner},
    title        = {learn2learn},
    month        = sep,
    year         = 2019,
    url          = {https://github.com/learnables/learn2learn}
    }
~~~

### Acknowledgements & Friends

1. The RL environments are adapted from Tristan Deleu's [implementations](https://github.com/tristandeleu/pytorch-maml-rl) and from the ProMP [repository](https://github.com/jonasrothfuss/ProMP/). Both shared with permission, under the MIT License.
2. [TorchMeta](https://github.com/tristandeleu/pytorch-meta) is similar library, with a focus on datasets for supervised meta-learning. 
3. [higher](https://github.com/facebookresearch/higher) is a PyTorch library that enables differentiating through optimization inner-loops. While they monkey-patch `nn.Module` to be stateless, learn2learn retains the stateful PyTorch look-and-feel. For more information, refer to [their ArXiv paper](https://arxiv.org/abs/1910.01727).
