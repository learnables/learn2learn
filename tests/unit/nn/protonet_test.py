#!/usr/bin/env python3

import unittest
import torch
import learn2learn as l2l

from learn2learn.utils import accuracy

IMAGE_SHAPES = (1, 28, 28)
NUM_CLASSES = 5
NUM_SHOTS = 5
NOISE = 0.0


class PrototypicalClassifierTests(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_simple(self):
        for distance in ["euclidean", "cosine"]:
            for normalize in [True, False]:
                # Create some fake data
                X = []
                y = []
                for i in range(NUM_CLASSES):
                    images = torch.randn(1, *IMAGE_SHAPES).expand(
                        NUM_SHOTS, *IMAGE_SHAPES
                    )
                    labels = torch.ones(NUM_SHOTS).long()
                    X.append(images)
                    y.append(i * labels)
                X = torch.cat(X, dim=0)
                y = torch.cat(y)
                X_support = X + torch.randn_like(X) * NOISE
                X_query = X + torch.randn_like(X) * NOISE

                # Compute embeddings
                X_support = X_support.view(NUM_CLASSES * NUM_SHOTS, -1)
                X_query = X_query.view(NUM_CLASSES * NUM_SHOTS, -1)

                classifier = l2l.nn.PrototypicalClassifier(
                    support=X_support,
                    labels=y,
                    distance=distance,
                    normalize=normalize,
                )
                predictions = classifier(X_query)
                acc = accuracy(predictions, y)
                self.assertTrue(acc >= 0.95)

    def test_euclidean_distance(self):
        pass

    def test_cosine_distance(self):
        pass


if __name__ == "__main__":
    unittest.main()
