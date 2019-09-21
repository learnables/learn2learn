# Meta-Learning & Computer Vision

This directory contains meta-learning examples and reproductions for common computer vision benchmarks.

### maml_omniglot.py

This file reproduces MAML on the Omniglot dataset.
The CNN results can be obtained by swapping `OmniglotFC` with `OmniglotCNN`.
As far as we know, the hyper-parameters should also replicate the 5-ways 5-shots setting.

Note that the original MAML paper trains with 1 fast adaptation step, but tests with 3 steps.
This implementation only provides the training code.

**Results**

| Setup                         | Original      | learn2learn  |
| ----------------------------- | ------------- | ------------ |
| FC, 1 shot, 5 ways, 1 step    | 89.7%         | 88.9%        |
| CNN, 1 shot, 5 ways, 1 step   | 98.7%         | 99.1%        |

**Usage**

~~~shell
python examples/vision/maml_omniglot.py
~~~

**Colab** Follow [this link](https://colab.research.google.com/drive/1N1vtHAPJBaJO1wD30b_fnwPOWDorm_gP) to see this example live on Google Colab.

### maml_miniimagenet.py

This file reproduces MAML on the *mini*-ImageNet dataset.
The FOMAML results can be obtained by setting `first_order=True` in the `MAML` wrapper.

Note that the original MAML paper trains with 5 fast adaptation step, but tests with 10 steps.
This implementation only provides the training code.

**Results**

| Setup                         | Original      | learn2learn  |
| ----------------------------- | ------------- | ------------ |
| CNN, 1 shot, 5 ways, 5 step   | 48.7%         | 48.3%        |
| CNN, 5 shot, 5 ways, 5 step   | 63.1%         | 64.8%        |

**Usage**

~~~shell
python examples/vision/maml_miniimagenet.py
~~~

**Colab** Follow [this link](https://colab.research.google.com/drive/1NIWbMYM3CercnoNMmlhre7hsJ8wMtAry) to see this example live on Google Colab.
