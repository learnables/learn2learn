#!/usr/bin/env python3

import random
from torchvision.transforms import RandomRotation


class RandomDiscreteRotation(RandomRotation):

    def __init__(self, degrees, *args, **kwargs):
        super(RandomDiscreteRotation, self).__init__(degrees[0], *args, **kwargs)
        self.degrees = degrees

    @staticmethod
    def get_params(degrees):
        angle = random.choice(degrees)
        return angle
