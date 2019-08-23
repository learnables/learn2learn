#!/usr/bin/env python3

import torch as th
from torch import nn

from scipy.stats import truncnorm


def truncated_normal_(tensor, mean=0.0, std=1.0):
    # PT doesn't have truncated normal.
    # https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/18
    values = truncnorm.rvs(-2, 2, size=tensor.shape)
    values = mean + std * values
    tensor.copy_(th.from_numpy(values))
    return tensor


def maml_fc_init_(module):
    if hasattr(module, 'weight') and module.weight is not None:
        truncated_normal_(module.weight.data, mean=0.0, std=0.01)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias.data, 0.0)
    return module


class MAMLLinearBlock(nn.Module):

    def __init__(self, input_size, output_size):
        super(MAMLLinearBlock, self).__init__()
        self.relu = nn.ReLU()
        self.normalize = nn.BatchNorm1d(output_size,
                                        affine=True,
                                        momentum=0.999,
                                        eps=1e-3,
                                        track_running_stats=False,
                                        )
        # TODO: Remove affine and use AddBias
        # self.bias = AddBias(output_size)
        self.linear = nn.Linear(input_size, output_size)
        maml_fc_init_(self.linear)

    def forward(self, x):
        x = self.linear(x)
        # x = self.bias(x)
        x = self.normalize(x)
        x = self.relu(x)
        return x

class MAMLFC(nn.Sequential):

    def __init__(self, input_size, output_size, sizes=None):
        if sizes is None:
            sizes = [256, 128, 64, 64]
        layers = [MAMLLinearBlock(input_size, sizes[0]), ]
        for s_i, s_o in zip(sizes[:-1], sizes[1:]):
            layers.append(MAMLLinearBlock(s_i, s_o))
        layers.append(maml_fc_init_(nn.Linear(sizes[-1], output_size)))
        super(MAMLFC, self).__init__(*layers)
#        super(MAMLFC, self).__init__(
#            MAMLLinearBlock(input_size, 256),
#            MAMLLinearBlock(256, 128),
#            MAMLLinearBlock(128, 64),
#            MAMLLinearBlock(64, 64),
#            maml_fc_init_(nn.Linear(64, output_size)),
#        )
        self.input_size = input_size

    def forward(self, x):
        return super(MAMLFC, self).forward(x.view(-1, self.input_size))


def main(args=None):
    model = MAMLFC(28**2, 5)
    from PIL.Image import LANCZOS
    from torchvision.datasets import Omniglot
    from torchvision import transforms
    omniglot = Omniglot(root='./data',
                        background=True,
                        transform=transforms.Compose([
                            transforms.Resize(28, interpolation=LANCZOS),
                            transforms.ToTensor(),
                            lambda x: 1.0 - x,
                            # TODO: Add DiscreteRotations([0, 90, 180, 270])
                        ]),
                        download=True)
    


if __name__ == '__main__':
    main()
