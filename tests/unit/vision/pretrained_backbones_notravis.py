#!/usr/bin/env python3

import unittest
import torch
import learn2learn as l2l
import tempfile


class PretrainedBackboneTests(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_scale(self):
        with tempfile.TemporaryDirectory() as temp_path:
            for dataset in ['cifar-fs', 'fc100']:
                backbone = l2l.vision.models.get_pretrained_backbone(
                    model='cnn4',
                    dataset=dataset,
                    spec='default',
                    root=temp_path,
                    download=True,
                )
            for dataset in ['mini-imagenet', 'tiered-imagenet']:
                for model in ['resnet12', 'wrn28']:
                    backbone = l2l.vision.models.get_pretrained_backbone(
                        model=model,
                        dataset=dataset,
                        spec='default',
                        root=temp_path,
                        download=True,
                    )


if __name__ == "__main__":
    unittest.main()
