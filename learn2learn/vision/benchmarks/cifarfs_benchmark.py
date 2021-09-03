#!/usr/bin/env python3

import torchvision as tv
import learn2learn as l2l

from learn2learn.data.transforms import NWays, KShots, LoadData, RemapLabels, ConsecutiveLabels


def cifarfs_tasksets(
    train_ways=5,
    train_samples=10,
    test_ways=5,
    test_samples=10,
    root='~/data',
    device=None,
    **kwargs,
):
    """Tasksets for CIFAR-FS benchmarks."""
    data_transform = tv.transforms.ToTensor()
    train_dataset = l2l.vision.datasets.CIFARFS(root=root,
                                                transform=data_transform,
                                                mode='train',
                                                download=True)
    valid_dataset = l2l.vision.datasets.CIFARFS(root=root,
                                                transform=data_transform,
                                                mode='validation',
                                                download=True)
    test_dataset = l2l.vision.datasets.CIFARFS(root=root,
                                               transform=data_transform,
                                               mode='test',
                                               download=True)
    if device is not None:
        train_dataset = l2l.data.OnDeviceDataset(
            dataset=train_dataset,
            device=device,
        )
        valid_dataset = l2l.data.OnDeviceDataset(
            dataset=valid_dataset,
            device=device,
        )
        test_dataset = l2l.data.OnDeviceDataset(
            dataset=test_dataset,
            device=device,
        )
    train_dataset = l2l.data.MetaDataset(train_dataset)
    valid_dataset = l2l.data.MetaDataset(valid_dataset)
    test_dataset = l2l.data.MetaDataset(test_dataset)

    train_transforms = [
        NWays(train_dataset, train_ways),
        KShots(train_dataset, train_samples),
        LoadData(train_dataset),
        RemapLabels(train_dataset),
        ConsecutiveLabels(train_dataset),
    ]
    valid_transforms = [
        NWays(valid_dataset, test_ways),
        KShots(valid_dataset, test_samples),
        LoadData(valid_dataset),
        ConsecutiveLabels(valid_dataset),
        RemapLabels(valid_dataset),
    ]
    test_transforms = [
        NWays(test_dataset, test_ways),
        KShots(test_dataset, test_samples),
        LoadData(test_dataset),
        RemapLabels(test_dataset),
        ConsecutiveLabels(test_dataset),
    ]

    _datasets = (train_dataset, valid_dataset, test_dataset)
    _transforms = (train_transforms, valid_transforms, test_transforms)
    return _datasets, _transforms
