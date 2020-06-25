# Meta-Optimization

This directory contains examples of using learn2learn for meta-optimization or meta-descent.

# Hypergradient

The script `hypergrad_mnist.py` demonstrates how to implement a slightly modified version of "[Online Learning Rate Adaptation with Hypergradient Descent](https://arxiv.org/abs/1703.04782)".
The implementation departs from the algorithm presented in the paper in two ways.

1. We forgo the analytical formulation of the learning rate's gradient to demonstrate the capability of the `LearnableOptimizer` class.
2. We adapt per-parameter learning rates instead of updating a single learning rate shared by all parameters.

**Usage**

!!! warning
    The parameters for this script were not carefully tuned.

Manually edit the script and run:

~~~shell
python examples/optimization/hypergrad_mnist.py
~~~
