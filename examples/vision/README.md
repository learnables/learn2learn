# Meta-Learning & Computer Vision

This directory contains meta-learning examples and reproductions for common computer vision benchmarks.

## MAML

The following files reproduce MAML on the Omniglot and *mini*-ImageNet datasets.
The FOMAML results can be obtained by setting `first_order=True` in the `MAML` wrapper.
On Omniglot, the CNN results can be obtained by swapping `OmniglotFC` with `OmniglotCNN`.

* [maml_omniglot.py](https://github.com/learnables/learn2learn/blob/master/examples/vision/maml_omniglot.py) - MAML on the Omniglot dataset with a fully-connected network.
* [maml_miniimagenet.py](https://github.com/learnables/learn2learn/blob/master/examples/vision/maml_miniimagenet.py) - MAML on the *mini*-ImageNet dataset with the standard convolutional network.

Note that the original MAML paper trains with 5 fast adaptation step, but tests with 10 steps.
This implementation only provides the training code.

When adapting the code to different datasets, we obtained the following results.
Only the fast-adaptation learning rate needs a bit of tuning, and good values usually lie in a 0.5-2x range of the original value.

| Dataset       | Architecture | Ways | Shots | Original | learn2learn |
|---------------|--------------|------|-------|----------|-------------|
| Omniglot      | FC           | 5    | 1     | 89.7%    | 88.9%       |
| Omniglot      | CNN          | 5    | 1     | 98.7%    | 99.1%       |
| mini-ImageNet | CNN          | 5    | 1     | 48.7%    | 48.3%       |
| mini-ImageNet | CNN          | 5    | 5     | 63.1%    | 65.4%       |
| CIFAR-FS      | CNN          | 5    | 5     | 71.5%    | 73.6%       |
| FC100         | CNN          | 5    | 5     | n/a      | 49.0%       |

**Usage**

Manually edit the respective files and run:

~~~shell
python examples/vision/maml_omniglot.py
~~~

or

~~~shell
python examples/vision/maml_miniimagenet.py
~~~

## Prototypical Networks

The file [protonet_miniimagenet.py](https://github.com/learnables/learn2learn/blob/master/examples/vision/protonet_miniimagenet.py) reproduces protonet on the *mini*-ImageNet dataset.

This implementation provides training and testing code.


| Dataset       | Architecture | Ways | Shots | Original | learn2learn |
|---------------|--------------|------|-------|----------|-------------|
| mini-ImageNet | CNN          | 5    | 1     | 49.4%    | 49.1%       |
| mini-ImageNet | CNN          | 5    | 5     | 68.2%    | 66.5%       |


**Usage**

For 1 shot 5 ways:

~~~shell
python examples/vision/protonet_miniimagenet.py
~~~

For 5 shot 5 ways:

~~~shell
python examples/vision/protonet_miniimagenet.py --shot 5 --train-way 20
~~~

## ANIL

The file [anil_fc100.py](https://github.com/learnables/learn2learn/blob/master/examples/vision/anil_fc100.py) implements ANIL on the FC100 dataset.

While ANIL only used *mini*-ImageNet as a benchmark, we provide results for CIFAR-FS and FC100 as well.


| Dataset       | Architecture | Ways | Shots | Original | learn2learn |
|---------------|--------------|------|-------|----------|-------------|
| mini-ImageNet | CNN          | 5    | 5     | 61.5%    | 63.2%       |
| CIFAR-FS      | CNN          | 5    | 5     | n/a      | 68.3%       |
| FC100         | CNN          | 5    | 5     | n/a      | 47.6%       |


**Usage**

Manually edit the above file and run:

~~~shell
python examples/vision/protonet_miniimagenet.py
~~~

## SimpleShot

