#!/usr/bin/env python3

import torch
try:
    from qpth.qp import QPFunction
except ImportError:
    from learn2learn.utils import _ImportRaiser
    QPFunction = _ImportRaiser('qpth', 'pip install qpth')

EPS = 1e-8


def kronecker(A, B):
    return torch.einsum("ab,cd->acbd", A, B).view(
        A.size(0) * B.size(0), A.size(1) * B.size(1)
    )


def onehot(x, dim):
    size = x.size(0)
    x = x.long()
    onehot = torch.zeros(size, dim, device=x.device)
    onehot.scatter_(1, x.view(-1, 1), 1.0)
    return onehot


def svm_logits(query, support, labels, ways, shots, C_reg=0.1, max_iters=15):
    num_support = support.size(0)
    num_query = query.size(0)
    device = support.device
    kernel = support @ support.t()
    I_ways = torch.eye(ways).to(device)
    block_kernel = kronecker(kernel, I_ways)
    block_kernel.add_(torch.eye(ways * num_support, device=device))
    labels_onehot = onehot(labels, dim=ways).view(1, -1).to(device)
    I_sw = torch.eye(num_support * ways, device=device)
    I_s = torch.eye(num_support, device=device)
    h = C_reg * labels_onehot
    A = kronecker(I_s, torch.ones(1, ways, device=device))
    b = torch.zeros(1, num_support, device=device)
    qp = QPFunction(verbose=False, maxIter=max_iters)
    qp_solution = qp(block_kernel, -labels_onehot, I_sw, h, A, b)
    qp_solution = qp_solution.reshape(num_support, ways)

    qp_solution = qp_solution.unsqueeze(1).expand(num_support, num_query, ways)
    compatibility = support @ query.t()
    compatibility = compatibility.unsqueeze(2).expand(num_support, num_query, ways)
    logits = qp_solution * compatibility
    logits = torch.sum(logits, dim=0)
    return logits


class SVClassifier(torch.nn.Module):

    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/nn/metaoptnet.py)

    **Description**

    A module for the differentiable SVM classifier of MetaOptNet.

    **Arguments**

    * **support** (Tensor, *optional*, default=None) - Tensor of support features.
    * **labels** (Tensor, *optional*, default=None) - Labels corresponding to the support features.
    * **ways** (str, *optional*, default=None) - Number of classes in the task.
    * **normalize** (bool, *optional*, default=False) - Whether to normalize the inputs.
    * **C_reg** (float, *optional*, default=0.1) - Regularization weight for SVM.
    * **max_iters** (int, *optional*, default=15) - Maximum number of iterations for SVM convergence.

    **References**

    1. Lee et al. 2019. "Prototypical Networks for Few-shot Learning"

    **Example**

    ~~~python
    classifier = SVMClassifier()
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
        ways=None,
        normalize=False,
        C_reg=0.1,
        max_iters=15,
    ):
        super(SVClassifier, self).__init__()
        self.C_reg = C_reg
        self.max_iters = max_iters
        self._normalize = normalize
        if support is not None and labels is not None:
            if ways is None:
                ways = len(torch.unique(labels))
            self.fit_(support, labels, ways)

    def fit_(self, support, labels, ways=None, C_reg=None, max_iters=None):
        if C_reg is None:
            C_reg = self.C_reg
        if max_iters is None:
            max_iters = self.max_iters
        if self._normalize:
            support = self.normalize(support)
        if ways is None:
            ways = len(torch.unique(labels))
        num_support = support.size(0)
        device = support.device
        kernel = support @ support.t()
        I_ways = torch.eye(ways).to(device)
        block_kernel = kronecker(kernel, I_ways)
        block_kernel.add_(torch.eye(ways * num_support, device=device))
        labels_onehot = onehot(labels, dim=ways).view(1, -1).to(device)
        I_sw = torch.eye(num_support * ways, device=device)
        I_s = torch.eye(num_support, device=device)
        h = C_reg * labels_onehot
        A = kronecker(I_s, torch.ones(1, ways, device=device))
        b = torch.zeros(1, num_support, device=device)
        qp = QPFunction(verbose=False, maxIter=max_iters)
        qp_solution = qp(block_kernel, -labels_onehot, I_sw, h, A, b)
        self.qp_solution = qp_solution.reshape(num_support, ways)
        self.support = support
        self.num_support = num_support
        self.ways = ways

    @staticmethod
    def normalize(x, epsilon=EPS):
        x = x / (x.norm(p=2, dim=1, keepdim=True) + epsilon)
        return x

    def forward(self, x):
        if self._normalize:
            x = self.normalize(x)
        num_query = x.size(0)
        qp_solution = self.qp_solution.unsqueeze(1).expand(
            self.num_support,
            num_query,
            self.ways,
        )
        compatibility = self.support @ x.t()
        compatibility = compatibility.unsqueeze(2).expand(
            self.num_support,
            num_query,
            self.ways,
        )
        logits = qp_solution * compatibility
        logits = torch.sum(logits, dim=0)
        return logits


if __name__ == "__main__":
    from learn2learn.utils import accuracy

    IMAGE_SHAPES = (1, 16, 16)
    NUM_CLASSES = 10
    NUM_SHOTS = 5
    NOISE = 0.0

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
        X = X.cuda()
        y = y.cuda()
        X_support = X + torch.randn_like(X) * NOISE
        X_query = X + torch.randn_like(X) * NOISE

        # Compute embeddings
        X_support = X_support.view(NUM_CLASSES * NUM_SHOTS, -1)
        X_query = X_query.view(NUM_CLASSES * NUM_SHOTS, -1)

        classifier = SVClassifier(
            support=X_support,
            labels=y,
            normalize=normalize,
        )
        predictions = classifier(X_query)
        acc = accuracy(predictions, y)
        assert acc >= 0.95
