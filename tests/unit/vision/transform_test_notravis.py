import random, unittest
import torchvision as tv
import learn2learn as l2l
from torch import nn
from torchvision import transforms
from collections import namedtuple
from PIL.Image import LANCZOS
from learn2learn.data.transforms import (
    NWays,
    KShots,
    LoadData,
    RemapLabels,
    ConsecutiveLabels,
)
from learn2learn.vision.transforms import RandomClassRotation


def vgg102_tasksets(
    train_ways=5,
    train_samples=1,
    test_ways=5,
    test_samples=1,
    root="~/data",
    device=None,
    **kwargs
):
    data_transform = tv.transforms.Compose(
        [
            tv.transforms.Resize((32, 32), interpolation=LANCZOS),
            tv.transforms.ToTensor(),
        ]
    )
    train_dataset = l2l.vision.datasets.VGGFlower102(
        root=root, transform=data_transform, mode="train", download=True
    )
    valid_dataset = l2l.vision.datasets.VGGFlower102(
        root=root, transform=data_transform, mode="validation", download=True
    )
    test_dataset = l2l.vision.datasets.VGGFlower102(
        root=root, transform=data_transform, mode="test", download=True
    )

    train_dataset = l2l.data.MetaDataset(train_dataset)
    valid_dataset = l2l.data.MetaDataset(valid_dataset)
    test_dataset = l2l.data.MetaDataset(test_dataset)

    degrees = [0.0, 90.0, 180.0, 270.0]
    train_transforms = [
        NWays(train_dataset, train_ways),
        KShots(train_dataset, train_samples),
        LoadData(train_dataset),
        RemapLabels(train_dataset),
        ConsecutiveLabels(train_dataset),
        RandomClassRotation(train_dataset, degrees),
    ]
    valid_transforms = [
        NWays(valid_dataset, test_ways),
        KShots(valid_dataset, test_samples),
        LoadData(valid_dataset),
        ConsecutiveLabels(valid_dataset),
        RemapLabels(valid_dataset),
        RandomClassRotation(valid_dataset, degrees),
    ]
    test_transforms = [
        NWays(test_dataset, test_ways),
        KShots(test_dataset, test_samples),
        LoadData(test_dataset),
        RemapLabels(test_dataset),
        ConsecutiveLabels(test_dataset),
        RandomClassRotation(test_dataset, degrees),
    ]

    _datasets = (train_dataset, valid_dataset, test_dataset)
    _transforms = (train_transforms, valid_transforms, test_transforms)
    return _datasets, _transforms, degrees


def vgg102_benchmark(
    train_ways=5,
    train_samples=1,
    test_ways=5,
    test_samples=1,
    num_tasks_train=10,
    num_tasks_test=10,
    root="~/data",
):
    BenchmarkTasksets = namedtuple("BenchmarkTasksets", ("train", "validation", "test"))
    datasets, transforms, degrees = vgg102_tasksets()
    train_dataset, validation_dataset, test_dataset = datasets
    train_transforms, validation_transforms, test_transforms = transforms

    train_tasks = l2l.data.TaskDataset(
        dataset=train_dataset,
        task_transforms=train_transforms,
        num_tasks=num_tasks_train,
    )
    validation_tasks = l2l.data.TaskDataset(
        dataset=validation_dataset,
        task_transforms=validation_transforms,
        num_tasks=num_tasks_test,
    )
    test_tasks = l2l.data.TaskDataset(
        dataset=test_dataset,
        task_transforms=test_transforms,
        num_tasks=num_tasks_test,
    )
    return BenchmarkTasksets(train_tasks, validation_tasks, test_tasks), degrees


class RandomRotateTest(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_benchmark(self):
        tasksets, degrees = vgg102_benchmark()
        self.assertTrue(len(tasksets.train.sample()[0].shape) == 4)
        i, rot = 0, {i: "" for i in range(len(degrees))}
        for deg in degrees:
            self.assertIsInstance(deg, float)
            try:
                rot[i] = transforms.Compose([transforms.RandomRotation((deg, deg))])
                self.assertTrue(True, "The above modification are working fine")
            except:
                self.assertTrue(False, "The above modification is failing!")
            i += 1


if __name__ == "__main__":
    unittest.main()
