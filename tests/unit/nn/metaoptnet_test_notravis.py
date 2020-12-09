#!/usr/bin/env python3

import unittest
import torch
import learn2learn as l2l

from learn2learn.utils import accuracy

IMAGE_SHAPES = (1, 16, 16)
NUM_CLASSES = 10
NUM_SHOTS = 5
NOISE = 0.0


class SVClassifierTests(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_simple(self):
        for normalize in [True, False]:
            X = []
            y = []
            for i in range(NUM_CLASSES):
                images = torch.randn(1, *IMAGE_SHAPES).expand(NUM_SHOTS, *IMAGE_SHAPES)
                labels = torch.ones(NUM_SHOTS).long()
                X.append(images)
                y.append(i * labels)
            X = torch.cat(X, dim=0)
            y = torch.cat(y)
            X.requires_grad = True
            X_support = X + torch.randn_like(X) * NOISE
            X_query = X + torch.randn_like(X) * NOISE

            # Compute embeddings
            X_support = X_support.view(NUM_CLASSES * NUM_SHOTS, -1)
            X_query = X_query.view(NUM_CLASSES * NUM_SHOTS, -1)

            classifier = l2l.nn.SVClassifier(
                support=X_support,
                labels=y,
                normalize=normalize,
            )
            predictions = classifier(X_query)
            acc = accuracy(predictions, y)
            self.assertTrue(acc >= 0.95)


if __name__ == "__main__":
    unittest.main()
