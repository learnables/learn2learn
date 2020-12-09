#!/usr/bin/env python3

import torch

EPS = 1e-8


def compute_prototypes(support, labels):
    classes = torch.unique(labels)
    prototypes = torch.zeros(
        classes.size(0),
        *support.shape[1:],
        device=support.device,
        dtype=support.dtype,
    )
    for i, cls in enumerate(classes):
        embeddings = support[labels == cls]
        prototypes[i].add_(embeddings.mean(dim=0))
    return prototypes


class PrototypicalClassifier(torch.nn.Module):

    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/nn/protonet.py)

    **Description**

    A module for the differentiable nearest neighbour classifier of Prototypical Networks.

    **Arguments**

    * **support** (Tensor, *optional*, default=None) - Tensor of support features.
    * **labels** (Tensor, *optional*, default=None) - Labels corresponding to the support features.
    * **distance** (str, *optional*, default='euclidean') - Distance metric between samples. ['euclidean', 'cosine']
    * **normalize** (bool, *optional*, default=False) - Whether to normalize the inputs. Defaults to True when distance='cosine'.

    **References**

    1. Snell et al. 2017. "Prototypical Networks for Few-shot Learning"

    **Example**

    ~~~python
    classifier = PrototypicalClassifier()
    support = features(support_data)
    classifier.fit_(support, labels)
    query = features(query_data)
    preds = classifier(query)
    ~~~
    """

    def __init__(
        self,
        support=None,
        labels=None,
        distance="euclidean",
        normalize=False,
    ):
        super(PrototypicalClassifier, self).__init__()
        self.distance = "euclidean"
        self.normalize = normalize

        # Select compute_prototypes function
        self._compute_prototypes = compute_prototypes

        # Assign distance function
        if distance == "euclidean":
            self.distance = PrototypicalClassifier.euclidean_distance
        elif distance == "cosine":
            self.distance = PrototypicalClassifier.cosine_distance
            self.normalize = True
        else:
            self.distance = distance

        # Compute prototypes
        self.prototypes = None
        if support is not None and labels is not None:
            self.fit_(support, labels)

    def fit_(self, support, labels):
        """
        **Description**

        Computes and updates the prototypes given support embeddings and
        corresponding labels.

        """
        # TODO: Make a differentiable version? (For Proto-MAML style algorithms)

        # Compute new prototypes
        prototypes = self._compute_prototypes(support, labels)

        # Normalize if necessary
        if self.normalize:
            prototypes = PrototypicalClassifier.normalize(prototypes)

        # Assign prototypes and return them
        self.prototypes = prototypes
        return prototypes

    @staticmethod
    def cosine_distance(prototypes, queries):
        # Assumes both prototypes and queries are already normalized
        return -torch.mm(queries, prototypes.t())

    @staticmethod
    def euclidean_distance(prototypes, queries):
        n = prototypes.size(0)
        m = queries.size(0)
        prototypes = prototypes.unsqueeze(0).expand(m, n, -1)
        queries = queries.unsqueeze(1).expand(m, n, -1)
        distance = (prototypes - queries).pow(2).sum(dim=2)
        return distance

    @staticmethod
    def normalize(x, epsilon=EPS):
        x = x / (x.norm(p=2, dim=1, keepdim=True) + epsilon)
        return x

    def forward(self, x):
        assert (
            self.prototypes is not None
        ), "Prototypes not computed, use compute_prototypes(support, labels)"
        if self.normalize:
            x = PrototypicalClassifier.normalize(x)
        return -self.distance(self.prototypes, x)


if __name__ == "__main__":

    def accuracy(preds, targets):
        """Computes accuracy"""
        acc = (preds.argmax(dim=1).long() == targets.long()).sum().float()
        return acc / preds.size(0)

    IMAGE_SHAPES = (1, 16, 16)
    NUM_CLASSES = 100
    NUM_SHOTS = 5
    NOISE = 0.0

    for distance in ["euclidean", "cosine"]:
        for normalize in [True, False]:
            # Create some fake data
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
            #            X = X.cuda()
            #            y = y.cuda()
            X_support = X + torch.randn_like(X) * NOISE
            X_query = X + torch.randn_like(X) * NOISE

            # Compute embeddings
            X_support = X_support.view(NUM_CLASSES * NUM_SHOTS, -1)
            X_query = X_query.view(NUM_CLASSES * NUM_SHOTS, -1)

            classifier = PrototypicalClassifier(
                support=X_support,
                labels=y,
                distance=distance,
                normalize=normalize,
            )
            predictions = classifier(X_query)
            acc = accuracy(predictions, y)
            assert acc >= 0.95
