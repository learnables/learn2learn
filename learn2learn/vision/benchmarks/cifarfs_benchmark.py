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
    data_augmentation=None,
    device=None,
    **kwargs,
):
    """Tasksets for CIFAR-FS benchmarks."""
    if data_augmentation is None:
        train_data_transforms = tv.transforms.ToTensor()
        test_data_transforms = tv.transforms.ToTensor()
    elif data_augmentation == 'normalize':
        train_data_transforms = Compose([
            lambda x: x / 255.0,
        ])
        test_data_transforms = train_data_transforms
    elif data_augmentation == 'rfs2020':
        """
        # original
    if augment:
        transform = transforms.Compose([
            lambda x: Image.fromarray(x),
            transforms.RandomCrop(32, padding=4),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.RandomHorizontalFlip(),
            lambda x: np.asarray(x),
            transforms.ToTensor(),
            normalize_cifar100
        ])
    else:
        transform = transforms.Compose([
            lambda x: Image.fromarray(x),
            transforms.ToTensor(),
            normalize_cifar100
        ])
    return transform
    """
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
        normalize = tv.transforms.Normalize(mean=mean, std=std)
        train_data_transforms = Compose([
            ToPILImage(),
            RandomCrop(32, padding=4),
            ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ])
        test_data_transforms = Compose([
            ToTensor(),
            normalize,
        ])
    else:
        raise('Invalid data_augmentation argument.')
        
    train_dataset = l2l.vision.datasets.CIFARFS(root=root,
                                                transform=train_data_transforms,
                                                mode='train',
                                                download=True)
    valid_dataset = l2l.vision.datasets.CIFARFS(root=root,
                                                transform=train_data_transforms,
                                                mode='validation',
                                                download=True)
    test_dataset = l2l.vision.datasets.CIFARFS(root=root,
                                               transform=test_data_transforms,
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
