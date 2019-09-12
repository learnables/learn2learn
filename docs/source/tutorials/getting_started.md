### Getting Started

L2L is a meta-learning library providing three levels of functionality for users.
At a high level, there are many examples using meta-learning algorithms to train
on a myriad of datasets/environments. At a mid level, it provides a functional
interface for several popular meta-learning algorithms as well as a data loader
to make it easier to import other data sets. At a low level, it provides extended
functionality for modules.


### What is meta-learning?

Machine learning is typically concerned with the process of adapting an agent
or model to perform well on a given task \( \mathcal{T} \). If any aspect of \( \mathcal{T} \)
changes then we must begin training anew; however, it is easy to imagine several
situations where we may want to teach an agent or train a model to perform several
tasks that are very similar in nature or skill required. In this case, we would like
to extract "general" knowledge from training on an individual task to reduce the
amount of time and data needed to train on a subsequent later task. To formalize
this notion we assume that the tasks trained on are *i.i.d.* samples \( \{\mathcal{T}_{1}
\dotsb \mathcal{T}_{m}\} \), and we have a loss function \( \mathcal{L} \) defined for all \(
\mathcal{T} \). We can then phrase the problem of meta-learning a few ways. One way is
as k-shot learning, where we aim to find a model or policy M that minimizes \( E_{\mathcal{T}
}[\mathcal{L}(M_{k}(\mathcal{T}))] \), where \( M_{k} \) denotes the model M after training on
\( \mathcal{T} \) for k episodes.



For more information about specific meta-learning algorithms, please refer to the
appropriate tutorial.

### How to Use L2L

### Installing

A pip package is available, updated periodically. Use the command:

```pip install learn2learn```

For the most update-to-date version clone the [repository](https://github.com/learnables/learn2learn) and use:

```pip install -e .```

A list of dependencies is maintained and periodically updated in requirements-dev.txt. To install them all use 

```pip install -r requirements-dev.txt```

**Important:** As learn2learn is still in the developmental stage, breaking changes are likely to occur. If you
encounter a problem, feel free to an open an [issue](https://github.com/learnables/learn2learn/issues) and we'll
look into it.

### Source Files

Examples of learn2learn in action can be found [here](https://github.com/learnables/learn2learn/tree/master/examples).
The source code for algorithm implementations is also available [here](https://github.com/learnables/learn2learn/tree/master/learn2learn/algorithms).

