#!/usr/bin/env python

import torch
import torch.nn.init as init
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1):
    return torch.nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=True,
    )


def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform(m.weight, gain=2**0.5)
        init.constant(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant(m.weight, 1)
        init.constant(m.bias, 0)


class wide_basic(torch.nn.Module):

    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = torch.nn.BatchNorm2d(in_planes)
        self.conv1 = torch.nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.bn2 = torch.nn.BatchNorm2d(planes)
        self.conv2 = torch.nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = torch.nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_planes,
                    planes,
                    kernel_size=1,
                    stride=stride,
                    bias=True
                ),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)
        return out


class WideResNet(torch.nn.Module):

    def __init__(self, depth, widen_factor, dropout_rate):
        super(WideResNet, self).__init__()
        self.in_planes = 16

        assert ((depth - 4) % 6 == 0), 'Wide-resnet depth should be 6n+4'
        n = int((depth - 4) / 6)
        k = widen_factor

        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3(3, nStages[0])
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = torch.nn.BatchNorm2d(nStages[3], momentum=0.9)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return torch.nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 21)
        out = out.view(out.size(0), -1)
        return out


class WRN28Backbone(WideResNet):

    def __init__(self, dropout=0.0):
        super(WRN28Backbone, self).__init__(
            depth=28,
            widen_factor=10,
            dropout_rate=dropout,
        )


class WRN28(torch.nn.Module):

    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/vision/models/wrn28.py)

    **Description**

    The 28-layer 10-depth wide residual network from Dhillon et al, 2020.

    The code is adapted from [Ye et al, 2020](https://github.com/Sha-Lab/FEAT)
    who share it under the MIT license.

    Instantiate `WRN28Backbone` if you only need the feature extractor.

    **References**

    1. Dhillon et al. 2020. “A Baseline for Few-Shot Image Classification.” ICLR 20.
    2. Ye et al. 2020. “Few-Shot Learning via Embedding Adaptation with Set-to-Set Functions.” CVPR 20.
    3. Ye et al's code: [https://github.com/Sha-Lab/FEAT](https://github.com/Sha-Lab/FEAT)

    **Arguments**

    * **output_size** (int) - The dimensionality of the output.
    * **hidden_size** (list, *optional*, default=640) - Size of the embedding once features are extracted.
        (640 is for mini-ImageNet; used for the classifier layer)
    * **dropout** (float, *optional*, default=0.0) - Dropout rate.

    **Example**
    ~~~python
    model = WRN28(output_size=ways, hidden_size=1600, avg_pool=False)
    ~~~
    """

    def __init__(
        self,
        output_size,
        hidden_size=640,  # default for mini-ImageNet
        dropout=0.0,
    ):
        super(WRN28, self).__init__()
        self.features = WRN28Backbone(dropout=dropout)
        self.classifier = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    wrn = WRN28(output_size=5)
    img = torch.randn(5, 3, 84, 84)
    out = wrn(img)
    wrnbb = WRN28Backbone()
    out = wrnbb(img)
