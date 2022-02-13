import random, unittest, torch
import torchvision as tv
import learn2learn as l2l
from torchvision import transforms
from learn2learn.vision.transforms import RandomClassRotation


class RandomDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels, rotations):
        super(RandomDataset, self).__init__()
        self.values = data
        self.labels = labels
        self.rot = rotations

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        sample = self.values[index], self.labels[index]
        if float(tv.__version__.split('.')[1]) >= 11:
            transform = tv.transforms.RandomRotation((self.rot, self.rot))
        else:
            transform = tv.transforms.Compose(
                [
                    tv.transforms.ToPILImage(),
                    tv.transforms.RandomRotation((self.rot, self.rot)),
                    tv.transforms.ToTensor(),
                ]
            )
        sample = transform(sample[0])
        return sample


class RandomRotateTest(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_benchmark(self):
        for _ in range(10):
            sample, channels, size = (
                random.randrange(500, 5000),
                random.randrange(1, 5),
                random.randrange(20, 200),
            )
            data, labels = torch.rand(sample, channels, size, size), torch.rand(
                sample, 1
            )
            degrees = [0.0, 90.0, 180.0, 270.0]
            rotation = random.choice(degrees)
            try:
                dataset = RandomDataset(data, labels, rotation)
            except:
                self.assertTrue(False, "The above modification is failing.")
            self.assertTrue(len(dataset[0].shape) == 3)
            self.assertIsInstance(rotation, float)
            self.assertTrue(dataset[0].shape[0] == channels)


if __name__ == "__main__":
    unittest.main()
