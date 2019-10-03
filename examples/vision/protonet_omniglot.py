import argparse

import numpy as np
import torch
from PIL.Image import LANCZOS
from torch.nn import Module
from torch.optim import Adam
from torch.optim import Optimizer
from torchvision import transforms
from typing import Callable
import random
import learn2learn as l2l
from learn2learn.vision.datasets.full_omniglot import FullOmniglot
from learn2learn.vision.models import OmniglotCNN


def lr_schedule(epoch, lr):
    # Drop lr every 2000 episodes
    if epoch % drop_lr_every == 0:
        return lr / 2
    else:
        return lr


def categorical_accuracy(y, y_pred):
    return torch.eq(y_pred.argmax(dim=-1), y).sum().item() / y_pred.shape[0]


def batch_metrics(model: Module, y_pred: torch.Tensor, y: torch.Tensor,
                  batch_logs: dict):
    """Calculates metrics for the current training batch
    # Arguments
        model: Model being fit
        y_pred: predictions for a particular batch
        y: labels for a particular batch
        batch_logs: Dictionary of logs for the current batch
    """
    model.eval()
    batch_logs['categorical_accuracy'] = categorical_accuracy(y, y_pred)

    return batch_logs


def gradient_step(model: Module, optimiser: Optimizer, loss_fn: Callable, x: torch.Tensor, y: torch.Tensor, **kwargs):
    model.train()
    optimiser.zero_grad()
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    loss.backward()
    optimiser.step()

    return loss, y_pred


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
    support = embeddings[:n_shot * k_way]
    queries = embeddings[n_shot * k_way:]
    prototypes = compute_prototypes(support, k_way, n_shot)
    distances = pairwise_distances(queries, prototypes, distance)

    log_p_y = (-distances).log_softmax(dim=1)
    loss = loss_fn(log_p_y, y)
    y_pred = (-distances).softmax(dim=1)

    if train:
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
        raise (ValueError('Unsupported similarity function'))


def set_lr(epoch, optimiser, lrs):
    for i, param_group in enumerate(optimiser.param_groups):
        new_lr = lrs[i]
        param_group['lr'] = new_lr
        print('Epoch {:5d}: setting learning rate'
              ' of group {} to {:.4e}.'.format(epoch, i, new_lr))
    return optimiser


def main(model: Module, optimiser: Optimizer, loss_fn: Callable, epochs: int,
         fit_function: Callable = gradient_step, fit_function_kwargs: dict = {}):
    batch_size = None

    print('Begin training...')

    monitor = f'val_{args.n_test}-shot_{args.k_test}-way_acc'
    monitor_op = np.less
    best = np.Inf
    epochs_since_last_save = 0
    epoch_logs = {}
    logs = epoch_logs or {}
    seen = 0
    metric_name = f"val_{args.n_test}-shot_{args.k_test}-way_acc"
    totals = {'loss': 0, metric_name: 0}
    for epoch in range(1, epochs + 1):
        lrs = [lr_schedule(epoch, param_group['lr'])
               for param_group in optimiser.param_groups]

        optimiser = set_lr(epoch, optimiser, lrs)

#         for batch_index, batch in enumerate(background_taskloader):
        for batch_index in range(training_episodes):
            batch_logs = dict(batch=batch_index, size=(batch_size or 1))
#             prep_batch = prepare_nshot_task(args.n_train, args.k_train, args.q_train)
#             x, y = prep_batch(batch)
            support_t_eval = train_generator.sample(shots=args.q_test)
            query_t_eval = train_generator.sample(
                shots=args.q_test, task=support_t_eval.sampled_task)
            x_support = torch.stack(support_t.data).double().to(device)
            y = torch.LongTensor(support_t.label).to(device)
            x_query = torch.stack(query_t.data).double().to(device)
            x = torch.cat([x_support, x_query], dim=0)
            loss, y_pred = fit_function(
                model, optimiser, loss_fn, x, y, **fit_function_kwargs)
            batch_logs['loss'] = loss.item()

            batch_logs = batch_metrics(model, y_pred, y, batch_logs)


#         for batch_index, batch_eval in enumerate(evaluation_taskloader):
#             prep_batch_eval = prepare_nshot_task(args.n_test, args.k_test, args.q_test)
#             x_eval, y_eval = prep_batch_eval(batch_eval)

        for batch_index in range(evaluation_episodes):
            support_t_eval = valid_generator.sample(shots=args.q_test)
            query_t_eval = valid_generator.sample(
                shots=args.q_test, task=support_t_eval.sampled_task)
            x_support_eval = torch.stack(
                support_t_eval.data).double().to(device)
            y_eval = torch.LongTensor(support_t_eval.label).to(device)
            x_query_eval = torch.stack(query_t_eval.data).double().to(device)

            x_eval = torch.cat([x_support_eval, x_query_eval], dim=0)
            loss, y_pred = proto_net_episode(
                model=model,
                optimiser=optimiser,
                loss_fn=loss_fn,
                x=x_eval,
                y=y_eval,
                n_shot=args.n_test,
                k_way=args.k_test,
                q_queries=args.q_test,
                train=False,
                distance=args.distance
            )

            seen += y_pred.shape[0]

            totals['loss'] += loss.item() * y_pred.shape[0]
            totals[metric_name] += categorical_accuracy(
                y_eval, y_pred) * y_pred.shape[0]

        logs['val_loss'] = totals['loss'] / seen
        logs[metric_name] = totals[metric_name] / seen
        print('Categorical Accuracy', logs[metric_name])
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

    print('Done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--distance', default='l2')
    parser.add_argument('--n-train', default=1, type=int)
    parser.add_argument('--n-test', default=1, type=int)
    parser.add_argument('--k-train', default=50, type=int)
    parser.add_argument('--k-test', default=5, type=int)
    parser.add_argument('--q-train', default=5, type=int)
    parser.add_argument('--q-test', default=1, type=int)
    args = parser.parse_args()

    training_episodes = 1000
    evaluation_episodes = 100
    n_epochs = 20000
    num_input_channels = 1
    drop_lr_every = 20

    assert torch.cuda.is_available()
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True

    param_str = f'omniglot_nt={args.n_train}_kt={args.k_train}_qt={args.q_train}_' \
        f'nv={args.n_test}_kv={args.k_test}_qv={args.q_test}'

    filepath = f'./data/{param_str}.pth'

    omniglot = FullOmniglot(root='./data',
                            transform=transforms.Compose([
                                l2l.vision.transforms.RandomDiscreteRotation(
                                    [0.0, 90.0, 180.0, 270.0]),
                                transforms.Resize(28, interpolation=LANCZOS),
                                transforms.ToTensor(),
                                lambda x: 1.0 - x,
                            ]),
                            download=True)
    omniglot = l2l.data.MetaDataset(omniglot)
    classes = list(range(1623))
    random.shuffle(classes)
    train_generator = l2l.data.TaskGenerator(dataset=omniglot,
                                             ways=args.k_train,
                                             classes=classes[:1100],
                                             tasks=20000)
    valid_generator = l2l.data.TaskGenerator(dataset=omniglot,
                                             ways=args.k_test,
                                             classes=classes[1100:1200],
                                             tasks=1024)
    test_generator = l2l.data.TaskGenerator(dataset=omniglot,
                                            ways=args.k_test,
                                            classes=classes[1200:],
                                            tasks=1024)

    model = OmniglotCNN()
    model.to(device, dtype=torch.double)

    optimiser = Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.NLLLoss().cuda()

    # test_generator = l2l.data.TaskGenerator(dataset=omniglot, ways=args.k_test, tasks=1024)
    # support_t = test_generator.sample(shots=args.q_test)
    # query_t = test_generator.sample(shots=args.q_test)

    main(
        model,
        optimiser,
        loss_fn,
        epochs=n_epochs,
        fit_function=proto_net_episode,
        fit_function_kwargs={'n_shot': args.n_train, 'k_way': args.k_train, 'q_queries': args.q_train, 'train': True,
                             'distance': args.distance},
    )