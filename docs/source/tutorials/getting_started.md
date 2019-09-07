# Getting Started

L2L is a meta-learning library providing three levels of functionality for users.
At a high level, there are many examples using meta-learning algorithms to train
on a myriad of datasets/environments. At a mid level, it provides a functional
interface for several popular meta-learning algorithms as well as a data loader
to make it easier to import other data sets. At a low level, it provides extended
functionality for modules.


# What is meta-learning?

Machine learning is typically concerned with the process of adapting an agent
or model to perform well on a given task \( \mathcal{T} \). If any aspect of \( \mathcal{T} \)
changes then we must begin training anew; however, it is easy to imagine several
situations where we may want to teach an agent or train a model to perform several
tasks that are very similar in nature or skill required. In this case, we would like
to extract "general" knowledge from training on an individual task to reduce the
amount of time and data needed to train on a subsequent later task. To formalize
this notion we assume that the tasks trained on are *i.i.d.* samples \( \{T_{1}
\dotsb T_{m}\} \) and we call their distribution p\( T \). Each task is
of the form \( T = \{ \mathcal{L}(x_{1},a_{1},\dotsb,x_{E},a_{E}),p(x_{1}),
p(x_{t+1} | x_{t},a_{t}),E \} \) with p() denoting a distribution and
E indicating the number of episodes (1 for supervised learning). We can then
phrase the problem of meta-learning a few ways. One way is as k-shot learning,
where we aim to find a model or policy M : \( (x_{1},\dotsb, x_{n})^{\intercal}
\rightarrow (a_{1},\dotsb,a_{n})^{\intercal}\) that minimizes \( E_{\mathcal{T}\sim
p(\mathcal{T})}[\mathcal{L}(\vec{x},M_{k}(\vec{x}))] \), where \( M_{k} \)
denotes the model M after training the task on k episodes.


For more information about specific meta-learning algorithms, please refer to the
appropriate tutorial.

# How to Use L2L
## Installing
## Source Files


