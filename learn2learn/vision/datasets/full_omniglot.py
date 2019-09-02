#!/usr/bin/env python3

from torch.utils.data import Dataset, ConcatDataset
from torchvision.datasets.omniglot import Omniglot


class FullOmniglot(Dataset):

    def __init__(self, root, transform=None, target_transform=None, download=False):
        self.transform = transform
        self.target_transform = target_transform

        # Set up both the background and eval dataset
        omni_background = Omniglot(root, background=True, download=download)
        # Eval labels also start from 0.
        # It's important to add 964 to label values in eval so they don't overwrite background dataset.
        omni_evaluation = Omniglot(root,
                                   background=False,
                                   download=download,
                                   target_transform=lambda x: x + len(omni_background._characters))

        self.dataset = ConcatDataset((omni_background, omni_evaluation))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        image, character_class = self.dataset[item]
        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            character_class = self.target_transform(character_class)

        return image, character_class
