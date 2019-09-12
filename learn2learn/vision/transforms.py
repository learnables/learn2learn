#!/usr/bin/env python3

"""

**Description**

A set of transformations commonly used in meta-learning vision tasks.
"""

import random

from torchvision.transforms import RandomRotation


class RandomDiscreteRotation(RandomRotation):
    """

    [[Source]]()

    **Description**

    Samples rotations from a given list, uniformly at random.

    **Arguments**

    * **degrees** (list) - The rotations to be sampled.

    **Example**
    ~~~python
    transform = RandomDiscreteRotation([0, 90, 180, 270])
    ~~~

    """

    def __init__(self, degrees, *args, **kwargs):
        super(RandomDiscreteRotation, self).__init__(degrees[0], *args, **kwargs)
        self.degrees = degrees

    @staticmethod
    def get_params(degrees):
        angle = random.choice(degrees)
        return angle
