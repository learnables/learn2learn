#!/usr/bin/env python3

"""

**Description**

A set of transformations commonly used in meta-learning vision tasks.
"""

import random
import torchvision as tv
from torchvision import transforms


class RandomClassRotation(object):
    """

    [[Source]]()

    **Description**

    Samples rotations from a given list uniformly at random, and applies it to
    all images from a given class.

    **Arguments**

    * **degrees** (list) - The rotations to be sampled.

    **Example**
    ~~~python
    transform = RandomClassRotation([0, 90, 180, 270])
    ~~~

    """

    def __init__(self, dataset, degrees):
        self.degrees = degrees
        self.dataset = dataset

    def __call__(self, task_description):
        rotations = {}
        for data_description in task_description:
            c = self.dataset.indices_to_labels[data_description.index]
            if c not in rotations:
                rot = random.choice(self.degrees)
                if float(tv.__version__.split('.')[1]) >= 11:
                    rotations[c] = transforms.RandomRotation((rot, rot))
                else:
                    rotations[c] = transforms.Compose(
                        [
                            transforms.ToPILImage(),
                            transforms.RandomRotation((rot, rot)),
                            transforms.ToTensor(),
                        ]
                    )
            rotation = rotations[c]
            data_description.transforms.append(lambda x: (rotation(x[0]), x[1]))
        return task_description
