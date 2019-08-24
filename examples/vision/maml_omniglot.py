#!/usr/bin/env python3

import random
import numpy as np

import torch as th
from torch import nn
from torch import optim

import learn2learn as l2l

from scipy.stats import truncnorm


def truncated_normal_(tensor, mean=0.0, std=1.0):
    # PT doesn't have truncated normal.
    # https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/18
    values = truncnorm.rvs(-2, 2, size=tensor.shape)
    values = mean + std * values
    tensor.copy_(th.from_numpy(values))
    return tensor


def maml_fc_init_(module):
    if hasattr(module, 'weight') and module.weight is not None:
        truncated_normal_(module.weight.data, mean=0.0, std=0.01)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias.data, 0.0)
    return module


class MAMLLinearBlock(nn.Module):

    def __init__(self, input_size, output_size):
        super(MAMLLinearBlock, self).__init__()
        self.relu = nn.ReLU()
        self.normalize = nn.BatchNorm1d(output_size,
                                        affine=True,
                                        momentum=0.999,
                                        eps=1e-3,
                                        track_running_stats=False,
                                        )
        self.linear = nn.Linear(input_size, output_size)
        maml_fc_init_(self.linear)

    def forward(self, x):
        x = self.linear(x)
        # x = self.bias(x)
        x = self.normalize(x)
        x = self.relu(x)
        return x

class MAMLFC(nn.Sequential):

    def __init__(self, input_size, output_size, sizes=None):
        if sizes is None:
            sizes = [256, 128, 64, 64]
        layers = [MAMLLinearBlock(input_size, sizes[0]), ]
        for s_i, s_o in zip(sizes[:-1], sizes[1:]):
            layers.append(MAMLLinearBlock(s_i, s_o))
        layers.append(maml_fc_init_(nn.Linear(sizes[-1], output_size)))
        super(MAMLFC, self).__init__(*layers)
#        super(MAMLFC, self).__init__(
#            MAMLLinearBlock(input_size, 256),
#            MAMLLinearBlock(256, 128),
#            MAMLLinearBlock(128, 64),
#            MAMLLinearBlock(64, 64),
#            maml_fc_init_(nn.Linear(64, output_size)),
#        )
        self.input_size = input_size

    def forward(self, x):
        return super(MAMLFC, self).forward(x.view(-1, self.input_size))


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


def fast_adapt(adaptation_data, evaluation_data, learner, loss, adaptation_steps, device):
    for step in range(adaptation_steps):
        data = [d for d in adaptation_data]
        X = th.cat([d[0] for d in data], dim=0).to(device)
        y = th.cat([th.tensor(d[1]).view(-1) for d in data], dim=0).to(device)
        train_error = loss(learner(X), y)
        train_error /= len(adaptation_data)
        learner.adapt(train_error)
    data = [d for d in evaluation_data]
    X = th.cat([d[0] for d in data], dim=0).to(device)
    y = th.cat([th.tensor(d[1]).view(-1) for d in data], dim=0).to(device)
    predictions = learner(X)
    valid_error = loss(predictions, y)
    valid_error /= len(evaluation_data)
    valid_accuracy = accuracy(predictions, y)
    return valid_error, valid_accuracy



def main(
        ways=5,
        shots=1,
        meta_lr=0.003,
        fast_lr=0.5,
        meta_batch_size=32,
        adaptation_steps=1,
        num_iterations=60000,
        cuda=True,
        seed=42,
    ):
    from PIL.Image import LANCZOS
    from torchvision.datasets import Omniglot
    from torchvision import transforms
    from torch.utils.data import ConcatDataset

    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    device = th.device('cpu')
    if cuda:
        th.cuda.manual_seed(seed)
        device = th.device('cuda')

    # Create Dataset
    # TODO: Create l2l.data.vision.FullOmniglot, which merges background and evaluation sets.
    omni_background = Omniglot(root='./data',
                               background=True,
                               transform=transforms.Compose([
                                   transforms.Resize(28, interpolation=LANCZOS),
                                   transforms.ToTensor(),
                                   # TODO: Add DiscreteRotations([0, 90, 180, 270])
                                   lambda x: 1.0 - x,
                               ]),
                               download=True)
#    max_y = 1 + max([y for X, y in omni_background])
    max_y = 964
    omni_evaluation = Omniglot(root='./data',
                               background=False,
                               transform=transforms.Compose([
                                   transforms.Resize(28, interpolation=LANCZOS),
                                   transforms.ToTensor(),
                                   # TODO: Add DiscreteRotations([0, 90, 180, 270])
                                   lambda x: 1.0 - x,
                               ]),
                               target_transform=transforms.Compose([
                                   lambda x: max_y + x,
                               ]),
                               download=True)
    omniglot = ConcatDataset((omni_background, omni_evaluation))
    train_generator = l2l.data.TaskGenerator(dataset=omniglot, ways=ways)
    valid_generator = l2l.data.TaskGenerator(dataset=omniglot, ways=ways)
    test_generator = l2l.data.TaskGenerator(dataset=omniglot, ways=ways)
    # TODO: Implement an easy way to split one dataset into splits, based on classes.

    # Create model
    model = MAMLFC(28**2, ways)
    model.to(device)
    maml = l2l.algorithms.MAML(model, lr=fast_lr, first_order=False)
    opt = optim.Adam(maml.parameters(), meta_lr)
    loss = nn.CrossEntropyLoss(size_average=True, reduction='mean')

    for iteration in range(num_iterations):
        opt.zero_grad()
        meta_train_error = 0.0
        meta_train_accuracy = 0.0
        meta_valid_error = 0.0
        meta_valid_accuracy = 0.0
        meta_test_error = 0.0
        meta_test_accuracy = 0.0
        for task in range(meta_batch_size):
            # Compute meta-training loss
            learner = maml.clone()
            adaptation_data = train_generator.sample(shots=shots)
            evaluation_data = train_generator.sample(shots=shots,
                                                     classes_to_sample=adaptation_data.sampled_classes)
            evaluation_error, evaluation_accuracy = fast_adapt(adaptation_data,
                                                               evaluation_data,
                                                               learner,
                                                               loss,
                                                               adaptation_steps,
                                                               device)
            evaluation_error.backward()
            meta_train_error += evaluation_error.item()
            meta_train_accuracy += evaluation_accuracy.item()

            # Compute meta-validation loss
            learner = maml.clone()
            adaptation_data = valid_generator.sample(shots=shots)
            evaluation_data = valid_generator.sample(shots=shots,
                                                     classes_to_sample=adaptation_data.sampled_classes)
            evaluation_error, evaluation_accuracy = fast_adapt(adaptation_data,
                                                               evaluation_data,
                                                               learner,
                                                               loss,
                                                               adaptation_steps,
                                                               device)
            meta_valid_error += evaluation_error.item()
            meta_valid_accuracy += evaluation_accuracy.item()

            # Compute meta-testing loss
            learner = maml.clone()
            adaptation_data = test_generator.sample(shots=shots)
            evaluation_data = test_generator.sample(shots=shots,
                                                    classes_to_sample=adaptation_data.sampled_classes)
            evaluation_error, evaluation_accuracy = fast_adapt(adaptation_data,
                                                               evaluation_data,
                                                               learner,
                                                               loss,
                                                               adaptation_steps,
                                                               device)
            meta_test_error += evaluation_error.item()
            meta_test_accuracy += evaluation_accuracy.item()

        # Print some metrics
        print('\n')
        print('Iteration', iteration)
        print('Meta Train Error', meta_train_error / meta_batch_size)
        print('Meta Train Accuracy', meta_train_accuracy / meta_batch_size)
        print('Meta Valid Error', meta_valid_error / meta_batch_size)
        print('Meta Valid Accuracy', meta_valid_accuracy / meta_batch_size)
        print('Meta Test Error', meta_test_error / meta_batch_size)
        print('Meta Test Accuracy', meta_test_accuracy / meta_batch_size)

        # Average the accumulated gradients and optimize
        for p in maml.parameters():
            p.grad.data.mul_(1.0 / meta_batch_size)
        opt.step()


if __name__ == '__main__':
    main()
