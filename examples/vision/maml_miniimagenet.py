#!/usr/bin/env python3

import random
import numpy as np

import torch as th
from torch import nn
from torch import optim

import learn2learn as l2l


def maml_init_(module):
    nn.init.xavier_uniform_(module.weight.data, gain=1.0)
    nn.init.constant_(module.bias.data, 0.0)
    return module


class MAMLConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, max_pool=True, max_pool_factor=1.0):
        super(MAMLConvBlock, self).__init__()
        stride = (int(2 * max_pool_factor), int(2 * max_pool_factor))
        if max_pool:
            self.max_pool = nn.MaxPool2d(kernel_size=stride,  # TODO: Correct ?
                                         stride=stride,
                                         ceil_mode=False,  # pad='VALID' (?)
                                         )
            stride = (1, 1)
        else:
            self.max_pool = lambda x: x
        self.normalize = nn.BatchNorm2d(out_channels,
                                        affine=True,
                                        eps=1e-3,
                                        momentum=0.999,
                                        track_running_stats=False,
                                        )
        self.relu = nn.ReLU()

        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size,
                              stride=stride,
                              padding=1,
                              bias=True)
        maml_init_(self.conv)

    def forward(self, x):
        x = self.conv(x)
        x = self.normalize(x)
        x = self.relu(x)
        x = self.max_pool(x)
        return x


class MAMLConvBase(nn.Sequential):

    """
    NOTE:
        Omniglot: hidden=64, channels=1, no max_pool
        MiniImagenet: hidden=32, channels=3, max_pool
    """

    def __init__(self, output_size, hidden=64, channels=1, max_pool=False, layers=4, mp_factor=1.0):
        core = [MAMLConvBlock(channels, hidden, (3, 3), max_pool=max_pool, max_pool_factor=mp_factor),]
        for l in range(layers - 1):
            core.append(MAMLConvBlock(hidden, hidden, (3, 3), max_pool=max_pool, max_pool_factor=mp_factor))
        super(MAMLConvBase, self).__init__(*core)



class MiniImageNetCNN(nn.Module):

    def __init__(self, output_size, hidden_size=32, layers=4):
        super(MiniImageNetCNN, self).__init__()
        self.base = MAMLConvBase(output_size=hidden_size,
                                 hidden=hidden_size,
                                 channels=3,
                                 max_pool=True,
                                 layers=layers,
                                 mp_factor=4//layers)
        self.linear = nn.Linear(25*hidden_size, output_size, bias=True)
        maml_init_(self.linear)
        self.hidden_size = hidden_size

    def forward(self, x):
        x = self.base(x)
        x = self.linear(x.view(-1, 25*self.hidden_size))
        return x



def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


def fast_adapt(adaptation_data, evaluation_data, learner, loss, adaptation_steps, device):
    for step in range(adaptation_steps):
        data = [d for d in adaptation_data]
        X = th.cat([d[0].unsqueeze(0) for d in data], dim=0).to(device)
        y = th.cat([th.tensor(d[1]).view(-1) for d in data], dim=0).to(device)
        train_error = loss(learner(X), y)
        train_error /= len(adaptation_data)
        learner.adapt(train_error)
    data = [d for d in evaluation_data]
    X = th.cat([d[0].unsqueeze(0) for d in data], dim=0).to(device)
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
    import os
    from PIL.Image import LANCZOS
    from torchvision.datasets import Omniglot, ImageFolder
    from torchvision import transforms
    from torch.utils.data import ConcatDataset
    from PIL import Image

    MI_PATH = '~/Dropbox/Temporary/mini-imagenet-l2l/miniimagenet/resized/'

    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    device = th.device('cpu')
    if cuda:
        th.cuda.manual_seed(seed)
        device = th.device('cuda')

    # Create Dataset
    transform = transforms.Compose([
#        lambda x: Image.open(x),
        transforms.ToTensor(),
        lambda x: x.float() / 255.,
    ])
    train_images_path = os.path.join(MI_PATH, 'train')
    valid_images_path = os.path.join(MI_PATH, 'val')
    test_images_path = os.path.join(MI_PATH, 'test')
    train_dataset = ImageFolder(train_images_path, transform)
    valid_dataset = ImageFolder(valid_images_path, transform)
    test_dataset = ImageFolder(test_images_path, transform)
    train_generator = l2l.data.TaskGenerator(dataset=train_dataset, ways=ways)
    valid_generator = l2l.data.TaskGenerator(dataset=valid_dataset, ways=ways)
    test_generator = l2l.data.TaskGenerator(dataset=test_dataset, ways=ways)

    # Create model
    model = MiniImageNetCNN(ways)
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
