#!/usr/bin/env python
# coding: utf-8

# The code is adapted from Oscar Knagg
# https://github.com/oscarknagg/few-shot
# and he has a great set of medium articles around it.

from torch.optim import Adam
from torch.utils.data import DataLoader
import argparse
import numpy as np
from torch import nn
from typing import List, Iterable, Callable, Tuple, Union
from torch.utils.data import Sampler
import torch
from torch.optim import Optimizer
from torch.nn import Module
from typing import Callable

import learn2learn as l2l
# from torchvision.datasets import Omniglot
from learn2learn.vision.datasets import FullOmniglot
from torchvision import transforms
from PIL.Image import LANCZOS
from torch.utils.data import ConcatDataset



assert torch.cuda.is_available()
device = torch.device('cuda')
torch.backends.cudnn.benchmark = True


##############
# Parameters #
##############
parser = argparse.ArgumentParser()
parser.add_argument('--dataset')
parser.add_argument('--distance', default='l2')
parser.add_argument('--n-train', default=1, type=int)
parser.add_argument('--n-test', default=1, type=int)
parser.add_argument('--k-train', default=60, type=int)
parser.add_argument('--k-test', default=5, type=int)
parser.add_argument('--q-train', default=5, type=int)
parser.add_argument('--q-test', default=1, type=int)
args = parser.parse_args()

evaluation_episodes = 100
n_epochs = 40
num_input_channels = 1
drop_lr_every = 20

param_str = f'{args.dataset}_nt={args.n_train}_kt={args.k_train}_qt={args.q_train}_' \
            f'nv={args.n_test}_kv={args.k_test}_qv={args.q_test}'
print(param_str)

filepath=f'./data/{param_str}.pth'


def lr_schedule(epoch, lr):
    # Drop lr every 2000 episodes
    if epoch % drop_lr_every == 0:
        return lr / 2
    else:
        return lr



def batch_metrics(model: Module, y_pred: torch.Tensor, y: torch.Tensor, metrics: List[Union[str, Callable]],
                  batch_logs: dict):
    """Calculates metrics for the current training batch

    # Arguments
        model: Model being fit
        y_pred: predictions for a particular batch
        y: labels for a particular batch
        batch_logs: Dictionary of logs for the current batch
    """
    model.eval()
    for m in metrics:
        if isinstance(m, str):
            batch_logs[m] = NAMED_METRICS[m](y, y_pred)
        else:
            # Assume metric is a callable function
            batch_logs = m(y, y_pred)

    return batch_logs


def conv_block(in_channels: int, out_channels: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
def gradient_step(model: Module, optimiser: Optimizer, loss_fn: Callable, x: torch.Tensor, y: torch.Tensor, **kwargs):
    model.train()
    optimiser.zero_grad()
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    loss.backward()
    optimiser.step()

    return loss, y_pred

    
def categorical_accuracy(y, y_pred):
    return torch.eq(y_pred.argmax(dim=-1), y).sum().item() / y_pred.shape[0]

NAMED_METRICS = {
    'categorical_accuracy': categorical_accuracy
}

# max_y = 964
# omni_background = Omniglot(root='./data',
#                                background=True,
#                                transform=transforms.Compose([
#                                    transforms.Resize(28, interpolation=LANCZOS),
#                                    transforms.ToTensor(),
#                                    # TODO: Add DiscreteRotations([0, 90, 180, 270])
#                                    lambda x: 1.0 - x,
#                                ]),
#                                download=True)

# omni_evaluation = Omniglot(root='./data',
#                                background=False,
#                                transform=transforms.Compose([
#                                    transforms.Resize(28, interpolation=LANCZOS),
#                                    transforms.ToTensor(),
#                                    # TODO: Add DiscreteRotations([0, 90, 180, 270])
#                                    lambda x: 1.0 - x,
#                                ]),
#                                target_transform=transforms.Compose([
#                                    lambda x: max_y + x,
#                                ]),
#                                download=True)
# omniglot = ConcatDataset((omni_background, omni_evaluation))

omniglot = l2l.vision.datasets.FullOmniglot(root='./data',
                                            transform=transforms.Compose([
                                               l2l.vision.transforms.RandomDiscreteRotation([0.0, 90.0, 180.0, 270.0]),
                                               transforms.Resize(28, interpolation=LANCZOS),
                                               transforms.ToTensor(),
                                               lambda x: 1.0 - x,
                                            ]),
                                            download=True)
omniglot = l2l.data.MetaDataset(omniglot)
#     classes = list(range(1623))
#     random.shuffle(classes)
#     train_generator = l2l.data.TaskGenerator(dataset=omniglot, ways=ways, classes=classes[:1100])
#     valid_generator = l2l.data.TaskGenerator(dataset=omniglot, ways=ways, classes=classes[1100:1200])
#     test_generator = l2l.data.TaskGenerator(dataset=omniglot, ways=ways, classes=classes[1200:])

model = nn.Sequential(
        conv_block(num_input_channels, 5),
        conv_block(5,5),
        conv_block(5,5),
        conv_block(5,5),
        Flatten(),
    )
model.to(device, dtype=torch.double)

optimiser = Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.NLLLoss().cuda()

eval_generator = l2l.data.TaskGenerator(dataset=omniglot, ways=args.k_test)
support_t = eval_generator.sample(shots=args.q_test)
query_t = eval_generator.sample(shots=args.q_test)

def proto_net_episode(model: Module,
                      optimiser: Optimizer,
                      loss_fn: Callable,
                      x: torch.Tensor,
                      y: torch.Tensor,
                      n_shot: int,
                      k_way: int,
                      q_queries: int,
                      distance: str,
                      train: bool):
    """Performs a single training episode for a Prototypical Network.

    # Arguments
        model: Prototypical Network to be trained.
        optimiser: Optimiser to calculate gradient step
        loss_fn: Loss function to calculate between predictions and outputs. Should be cross-entropy
        x: Input samples of few shot classification task
        y: Input labels of few shot classification task
        n_shot: Number of examples per class in the support set
        k_way: Number of classes in the few shot classification task
        q_queries: Number of examples per class in the query set
        distance: Distance metric to use when calculating distance between class prototypes and queries
        train: Whether (True) or not (False) to perform a parameter update

    # Returns
        loss: Loss of the Prototypical Network on this task
        y_pred: Predicted class probabilities for the query set on this task
    """
    if train:
        # Zero gradients
        model.train()
        optimiser.zero_grad()
    else:
        model.eval()


    embeddings = model(x)

    # Samples are ordered by the NShotWrapper class as follows:
    # k lots of n support samples from a particular class
    # k lots of q query samples from those classes
    support = embeddings[:n_shot*k_way]
    queries = embeddings[n_shot*k_way:]
    prototypes = compute_prototypes(support, k_way, n_shot)

    # Calculate squared distances between all queries and all prototypes
    # Output should have shape (q_queries * k_way, k_way) = (num_queries, k_way)
    distances = pairwise_distances(queries, prototypes, distance)

    # Calculate log p_{phi} (y = k | x)
    log_p_y = (-distances).log_softmax(dim=1)
    loss = loss_fn(log_p_y, y)

    # Prediction probabilities are softmax over distances
    y_pred = (-distances).softmax(dim=1)

    if train:
        # Take gradient step
        loss.backward()
        optimiser.step()
    else:
        pass

    return loss, y_pred


def compute_prototypes(support: torch.Tensor, k: int, n: int) -> torch.Tensor:
    """Compute class prototypes from support samples.

    # Arguments
        support: torch.Tensor. Tensor of shape (n * k, d) where d is the embedding
            dimension.
        k: int. "k-way" i.e. number of classes in the classification task
        n: int. "n-shot" of the classification task

    # Returns
        class_prototypes: Prototypes aka mean embeddings for each class
    """
    # Reshape so the first dimension indexes by class then take the mean
    # along that dimension to generate the "prototypes" for each class
#     print("Support shape is ", support.shape)
    class_prototypes = support.reshape(k, n, -1).mean(dim=1)
    
    return class_prototypes



def pairwise_distances(x: torch.Tensor,
                       y: torch.Tensor,
                       matching_fn: str) -> torch.Tensor:
    """Efficiently calculate pairwise distances (or other similarity scores) between
    two sets of samples.

    # Arguments
        x: Query samples. A tensor of shape (n_x, d) where d is the embedding dimension
        y: Class prototypes. A tensor of shape (n_y, d) where d is the embedding dimension
        matching_fn: Distance metric/similarity score to compute between samples
    """
    n_x = x.shape[0]
    n_y = y.shape[0]

    if matching_fn == 'l2':
        distances = (
                x.unsqueeze(1).expand(n_x, n_y, -1) -
                y.unsqueeze(0).expand(n_x, n_y, -1)
        ).pow(2).sum(dim=2)
        return distances
    elif matching_fn == 'cosine':
        normalised_x = x / (x.pow(2).sum(dim=1, keepdim=True).sqrt() + EPSILON)
        normalised_y = y / (y.pow(2).sum(dim=1, keepdim=True).sqrt() + EPSILON)

        expanded_x = normalised_x.unsqueeze(1).expand(n_x, n_y, -1)
        expanded_y = normalised_y.unsqueeze(0).expand(n_x, n_y, -1)

        cosine_similarities = (expanded_x * expanded_y).sum(dim=2)
        return 1 - cosine_similarities
    elif matching_fn == 'dot':
        expanded_x = x.unsqueeze(1).expand(n_x, n_y, -1)
        expanded_y = y.unsqueeze(0).expand(n_x, n_y, -1)

        return -(expanded_x * expanded_y).sum(dim=2)
    else:
        raise(ValueError('Unsupported similarity function'))


def set_lr(epoch, optimiser,lrs):
    for i, param_group in enumerate(optimiser.param_groups):
        new_lr = lrs[i]
        param_group['lr'] = new_lr
        print('Epoch {:5d}: setting learning rate'
              ' of group {} to {:.4e}.'.format(epoch, i, new_lr))
    return optimiser



def fit(model: Module, optimiser: Optimizer, loss_fn: Callable, epochs: int, 
        metrics: List[Union[str, Callable]] = None,
        verbose: bool =True, fit_function: Callable = gradient_step, fit_function_kwargs: dict = {}):

    # Determine number of samples:
    batch_size= None
    eval_dict = {
        'eval_fn':proto_net_episode,
        'num_tasks':evaluation_episodes,
        'n_shot':args.n_test,
        'k_way':args.k_test,
        'q_queries':args.q_test,
        'distance':args.distance,
        'prefix':'val_',
    }

    if verbose:
        print('Begin training...')
        
        
    train_generator = l2l.data.TaskGenerator(dataset=omniglot, ways=args.k_train)
    eval_generator = l2l.data.TaskGenerator(dataset=omniglot, ways=args.k_test)

    monitor=f'val_{args.n_test}-shot_{args.k_test}-way_acc'
    monitor_op = np.less
    best = np.Inf
    epochs_since_last_save = 0
    for epoch in range(1, epochs+1):
        lrs = [lr_schedule(epoch, param_group['lr']) for param_group in optimiser.param_groups]
        
        optimiser = set_lr(epoch, optimiser,lrs)
        
        epoch_logs = {}
        for batch_index in range(training_episodes):
            batch_logs = dict(batch=batch_index, size=(batch_size or 1))

            support_t = train_generator.sample(shots=args.q_train)
            query_t = train_generator.sample(shots=args.q_test)
            x_support = torch.stack(support_t.data).double().to(device)
            y_support = torch.LongTensor(support_t.label).to(device)
            x_query = torch.stack(query_t.data).double().to(device)
            x_support_query = torch.cat([x_support, x_query], dim=0)
            loss, y_pred = fit_function(model, optimiser, loss_fn, x_support_query, y_support, **fit_function_kwargs)
            batch_logs['loss'] = loss.item()

            batch_logs = batch_metrics(model, y_pred, y_support, metrics, batch_logs)

    
        
        logs = epoch_logs or {}
        seen = 0
        metric_name = f"{eval_dict['prefix']}{eval_dict['n_shot']}-shot_{eval_dict['k_way']}-way_acc"
        totals = {'loss': 0, metric_name: 0}
        for batch_index in range(evaluation_episodes):


            support_t_eval = eval_generator.sample(shots=args.q_test)
            query_t_eval = eval_generator.sample(shots=args.q_test)
            x_support_eval = torch.stack(support_t_eval.data).double().to(device)
            y_support_eval = torch.LongTensor(support_t_eval.label).to(device)
            x_query_eval = torch.stack(query_t_eval.data).double().to(device)

            x_support_query = torch.cat([x_support_eval, x_query_eval], dim=0)
            
            
            loss, y_pred = eval_dict['eval_fn'](
                model = model,
                optimiser= optimiser,
                loss_fn=loss_fn,
                x=x_support_query,
                y=y_support_eval,
                n_shot=eval_dict['n_shot'],
                k_way=eval_dict['k_way'],
                q_queries=eval_dict['q_queries'],
                train=False,
                distance=eval_dict['distance']
            )

            seen += y_pred.shape[0]

            totals['loss'] += loss.item() * y_pred.shape[0]
            totals[metric_name] += categorical_accuracy(y_support_eval, y_pred) * y_pred.shape[0]

        logs[eval_dict['prefix'] + 'loss'] = totals['loss'] / seen
        logs[metric_name] = totals[metric_name] / seen
        if len(optimiser.param_groups) == 1:
            logs['lr'] = optimiser.param_groups[0]['lr']
        else:    
            for i, param_group in enumerate(optimiser.param_groups):
                logs['lr_{}'.format(i)] = param_group['lr']
                
        epochs_since_last_save += 1
        current = logs.get(monitor)
        if np.less(current, best):
            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
          ' saving model to %s'
          % (epoch + 1, monitor, best,
             current, filepath))
            best = current
            torch.save(model.state_dict(), filepath)


    if verbose:
        print('Finished.')

fit(
    model,
    optimiser,
    loss_fn,
    epochs=n_epochs,
    metrics=['categorical_accuracy'],
    fit_function=proto_net_episode,
    fit_function_kwargs={'n_shot': args.n_train, 'k_way': args.k_train, 'q_queries': args.q_train, 'train': True,
                         'distance': args.distance},
)



