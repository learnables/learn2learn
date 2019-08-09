#!/usr/bin/env python3

import time

import torch
# import wandb
from torch import nn, optim
from torch.nn import functional as F
from tqdm import tqdm

import learn2learn as l2l

# wandb.init(project="learn2learn")

WAYS = 5
SHOTS = 1
TASKS_PER_STEPS = 32


class Net(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, num_classes, input_dim=768, inner_dim=200, pooler_dropout=0.3):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = nn.ReLU()
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, x, **kwargs):
        start = time.time()
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = F.log_softmax(self.out_proj(x), dim=1)
        return x


def accuracy(preds, targets):
    preds = preds.argmax(dim=1)
    acc = (preds == targets).sum().float()
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


def inner_training_loop(task, roberta, device, learner, loss_func, batch=15):
    loss = 0.0
    acc = 0.0
    for i, (X, y) in enumerate(torch.utils.data.DataLoader(
            task, batch_size=batch, shuffle=True, num_workers=0)):
        # RoBERTa ENCODING
        X = collate_tokens([roberta.encode(sent) for sent in X], pad_idx=1)
        with torch.no_grad():
            X = roberta.extract_features(X)
        X = X[:, 0, :]

        # Moving to device
        X, y = X.to(device), torch.tensor(y).view(-1).to(device)

        output = learner(X)
        curr_loss = loss_func(output, y)
        acc += accuracy(output, y)
        loss += curr_loss / len(task)
    loss /= len(task)
    return loss, acc


def main(file_location="/tmp/mnist"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    text_train = l2l.data.NewsClassification(root="/tmp/text", download=False, transform="roberta")
    train_gen = l2l.data.TaskGenerator(text_train, ways=WAYS)
    roberta = torch.hub.load('pytorch/fairseq', 'roberta.base')
    roberta.eval()
    roberta.to(device)
    model = Net(num_classes=WAYS)
    # for param in model.roberta.parameters():
    #     param.requires_grad = False

    model.to(device)
    meta_model = l2l.MAML(model, lr=0.01)
    opt = optim.Adam(meta_model.parameters(), lr=0.005)
    loss_func = nn.NLLLoss(reduction="sum")

    for iteration in tqdm(range(1000)):
        iteration_error = 0.0
        iteration_acc = 0.0
        for _ in range(TASKS_PER_STEPS):
            learner = meta_model.new()
            train_task = train_gen.sample(shots=SHOTS)
            valid_task = train_gen.sample(shots=SHOTS,
                                          classes_to_sample=train_task.sampled_classes)

            # Fast Adaptation
            for step in range(5):
                train_error, _ = inner_training_loop(train_task,
                                                     roberta,
                                                     device,
                                                     learner,
                                                     loss_func, batch=SHOTS * WAYS)
                learner.adapt(train_error)

            # Compute validation loss
            valid_error, valid_acc = inner_training_loop(valid_task, roberta, device, learner, loss_func,
                                                         batch=SHOTS * WAYS)
            iteration_error += valid_error
            iteration_acc += valid_acc

        iteration_error /= TASKS_PER_STEPS
        iteration_acc /= TASKS_PER_STEPS
        # wandb.log({"Validation error": iteration_error.item()})
        # wandb.log({"Validation accuracy": iteration_acc})
        print(iteration, 'Valid error:', iteration_error.item())
        print(iteration, 'Valid acc:', iteration_acc)

        # Take the meta-learning step
        opt.zero_grad()
        iteration_error.backward()
        opt.step()


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()
