<p align="center"><img src="https://raw.githubusercontent.com/learnables/learn2learn/gh-pages/assets/img/l2l-full.png" height="120px" /></p>

--------------------------------------------------------------------------------

[![Build Status](https://travis-ci.com/learnables/learn2learn.svg?branch=master)](https://travis-ci.com/learnables/learn2learn)

learn2learn is a PyTorch library for meta-learning implementations.

The goal of meta-learning is to enable agents to *learn how to learn*.
That is, we would like our agents to become better learners as they solve more and more tasks.
For example, the animation below shows an agent that learns to run after a only one parameter update.

<p align="center"><img src="http://learn2learn.net/assets/img/halfcheetah.gif" height="250px" /></p>

**Features**

learn2learn provides high- and low-level utilities for meta-learning.
The high-level utilities allow arbitrary users to take advantage of exisiting meta-learning algorithms.
The low-level utilities enable researchers to develop new and better meta-learning algorithms.

Some features of learn2learn include:

* Modular API: implement your own training loops with our low-level utilities.
* Provides various meta-learning algorithms (e.g. MAML, FOMAML, MetaSGD, ProtoNets, DiCE)
* Task generator with unified API, compatible with torchvision, torchtext, torchaudio, and cherry.
* Provides standardized meta-learning tasks for vision (Omniglot, mini-ImageNet), reinforcement learning (Particles, Mujoco), and even text (news classification).
* 100% compatible with PyTorch -- use your own modules, datasets, or libraries!

## Installation

~~~bash
pip install learn2learn
~~~

## API Demo

The following is an example of using the high-level MAML implementation on MNIST.
For more algorithms and lower-level utilities, please refer to the [documentation](http://learn2learn.net/docs/learn2learn/) or the [examples](https://github.com/learnables/learn2learn/tree/master/examples).

~~~python
import learn2learn as l2l

mnist = torchvision.datasets.MNIST(root="/tmp/mnist", train=True)

mnist = l2l.data.MetaDataset(mnist)
train_tasks = l2l.data.TaskDataset(mnist,
                                   task_transforms=[
                                        NWays(mnist, n=3),
                                        KShots(mnist, k=1),
                                        LoadData(mnist),
                                   ],
                                   num_tasks=10)
model = Net()
maml = l2l.algorithms.MAML(model, lr=1e-3, first_order=False)
opt = optim.Adam(maml.parameters(), lr=4e-3)

for iteration in range(num_iterations):
    learner = maml.clone()  # Creates a clone of model
    for task in train_tasks:
        # Split task in adaptation_task and evalutation_task
        # Fast adapt
        for step in range(adaptation_steps):
            error = compute_loss(adaptation_task)
            learner.adapt(error)

        # Compute evaluation loss
        evaluation_error = compute_loss(evaluation_task)

        # Meta-update the model parameters
        opt.zero_grad()
        evaluation_error.backward()
        opt.step()
~~~

## Changelog

A human-readable changelog is available in the [CHANGELOG.md](CHANGELOG.md) file.

## Documentation

Documentation and tutorials are available on learn2learn’s website: [http://learn2learn.net](http://learn2learn.net).

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
2. [TorchMeta](https://github.com/tristandeleu/pytorch-meta) is similar library, with a focus on supervised meta-learning. If learn2learn were missing a particular functionality, we would go check if TorchMeta has it. But we would also open an issue ;)
3. [higher](https://github.com/facebookresearch/higher) is a PyTorch library that also enables differentiating through optimization inner-loops. Their approach is different from learn2learn in that they monkey-patch nn.Module to be stateless. For more information, refer to [their ArXiv paper](https://arxiv.org/abs/1910.01727).
