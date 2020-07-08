#!/usr/bin/env python3

import learn2learn as l2l


def get_kronecker_dims(param):
    shape = param.shape
    if len(shape) == 2:  # FC
        n, m = shape
    elif len(shape) == 1:  # Bias
        n, m = shape[0], 1
    elif len(shape) == 4:  # CNN
        n = shape[1]
        m = shape[2] * shape[3]
    else:
        raise NotImplementedError('Layer not supported. Please open an issue.')
    return n, m


class KroneckerTransform(object):

    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/optim/transforms/module_transform.py)

    **Description**

    The KroneckerTransform creates a an optimization transform based on nn.Module's that admit a Kronecker factorization.
    (see `l2l.nn.Kronecker*`)

    Akin to the ModuleTransform, this class of transform instanciates a module from its class, based on a given parameter.
    But, instead of reshaping the gradients to shape `(1, param.numel())`, this class assumes a Kronecker factorization
    of the weights for memory and computational efficiency.

    The specific dimension of the Kronecker factorization depends on the the parameter's shape.
    For a weight of shape (n, m), a KroneckerLinear transform consists of two weights with shapes (n, n) and (m, m) rather
    than a single weight of shape (nm, nm).
    Refer to Arnold et al., 2019 for more details.

    **Arguments**

    * **kronecker_cls** (callable) - A callable that instantiates the Kronecker module used to transform gradients.

    **References**

    1. Arnold et al. 2019. "When MAML can adapt fast and how to assist when it cannot".

    **Example**
    ~~~python
    classifier = torch.nn.Linear(784, 10, bias=False)
    kronecker_transform = KroneckerTransform(l2l.nn.KroneckerLinear)
    kronecker_update = kronecker_transform(classifier.weight)
    loss(classifier(X), y).backward()
    update = kronecker_update(classifier.weight.grad)
    classifier.weight.data.add_(-lr, update)  # Not a differentiable update. See l2l.optim.DifferentiableSGD.
    ~~~
    """

    def __init__(self, kronecker_cls, bias=False, psd=True):
        self.kronecker_cls = kronecker_cls
        self.bias = bias
        self.psd = psd

    def __call__(self, param):
        """docstring for forward"""
        n, m = get_kronecker_dims(param)
        transform = self.kronecker_cls(
            n=n,
            m=m,
            bias=self.bias,
            psd=self.psd,
        )
        return l2l.optim.transforms.ReshapedTransform(
            transform=transform,
            shape=(-1, n, m)
        )
