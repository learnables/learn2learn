#!/usr/bin/env python3

import argparse
import random

import torch
from torch import nn, optim
from torch.nn import functional as F
from tqdm import tqdm

import learn2learn as l2l


class Net(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, num_classes, input_dim=768, inner_dim=200, pooler_dropout=0.3):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = nn.ReLU()
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, x, **kwargs):
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = F.log_softmax(self.out_proj(x), dim=1)
        return x


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1)
    acc = (predictions == targets).sum().float()
    acc /= len(targets)
    return acc.item()


def collate_tokens(values, pad_idx, eos_idx=None, left_pad=False, move_eos_to_beginning=False):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(v.size(0) for v in values)
    res = values[0].new(len(values), size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            assert src[-1] == eos_idx
            dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
    return res


def compute_loss(task, roberta, device, learner, loss_func, batch=15):
    loss = 0.0
    acc = 0.0
    for i, (x, y) in enumerate(torch.utils.data.DataLoader(
            task, batch_size=batch, shuffle=True, num_workers=0)):
        # RoBERTa ENCODING
        x = collate_tokens([roberta.encode(sent) for sent in x], pad_idx=1)
        with torch.no_grad():
            x = roberta.extract_features(x)
        x = x[:, 0, :]

        # Moving to device
        x, y = x.to(device), y.view(-1).to(device)

        output = learner(x)
        curr_loss = loss_func(output, y)
        acc += accuracy(output, y)
        loss += curr_loss / len(task)
    loss /= len(task)
    return loss, acc


def main(lr=0.005, maml_lr=0.01, iterations=1000, ways=5, shots=1, tps=32, fas=5, device=torch.device("cpu"),
         download_location="/tmp/text"):
    text_train = l2l.text.datasets.NewsClassification(root=download_location, download=True)
    train_gen = l2l.text.datasets.TaskGenerator(text_train, ways=ways)

    torch.hub.set_dir(download_location)
    roberta = torch.hub.load('pytorch/fairseq', 'roberta.base')
    roberta.eval()
    roberta.to(device)
    model = Net(num_classes=ways)
    model.to(device)
    meta_model = l2l.algorithms.MAML(model, lr=maml_lr)
    opt = optim.Adam(meta_model.parameters(), lr=lr)
    loss_func = nn.NLLLoss(reduction="sum")

    tqdm_bar = tqdm(range(iterations))
    for iteration in tqdm_bar:
        iteration_error = 0.0
        iteration_acc = 0.0
        for _ in range(tps):
            learner = meta_model.clone()
            train_task = train_gen.sample(shots=shots)
            valid_task = train_gen.sample(shots=shots, classes_to_sample=train_task.sampled_classes)

            # Fast Adaptation
            for step in range(fas):
                train_error, _ = compute_loss(train_task, roberta, device, learner, loss_func, batch=shots * ways)
                learner.adapt(train_error)

            # Compute validation loss
            valid_error, valid_acc = compute_loss(valid_task, roberta, device, learner, loss_func,
                                                  batch=shots * ways)
            iteration_error += valid_error
            iteration_acc += valid_acc

        iteration_error /= tps
        iteration_acc /= tps
        tqdm_bar.set_description("Loss : {:.3f} Acc : {:.3f}".format(iteration_error.item(), iteration_acc))

        # Take the meta-learning step
        opt.zero_grad()
        iteration_error.backward()
        opt.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Learn2Learn Text Classification Example')

    parser.add_argument('--ways', type=int, default=5, metavar='N',
                        help='number of ways (default: 5)')
    parser.add_argument('--shots', type=int, default=1, metavar='N',
                        help='number of shots (default: 1)')
    parser.add_argument('-tps', '--tasks-per-step', type=int, default=32, metavar='N',
                        help='tasks per step (default: 32)')
    parser.add_argument('-fas', '--fast-adaption-steps', type=int, default=5, metavar='N',
                        help='steps per fast adaption (default: 5)')

    parser.add_argument('--iterations', type=int, default=1000, metavar='N',
                        help='number of iterations (default: 1000)')

    parser.add_argument('--lr', type=float, default=0.005, metavar='LR',
                        help='learning rate (default: 0.005)')
    parser.add_argument('--maml-lr', type=float, default=0.01, metavar='LR',
                        help='learning rate for MAML (default: 0.01)')

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    parser.add_argument('--download-location', type=str, default="/tmp/text", metavar='S',
                        help='download location for train data and roberta(default : /tmp/text')

    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    main(lr=args.lr, maml_lr=args.maml_lr, iterations=args.iterations, ways=args.ways, shots=args.shots,
         tps=args.tasks_per_step, fas=args.fast_adaption_steps, device=device,
         download_location=args.download_location)
