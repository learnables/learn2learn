import argparse
import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import learn2learn as l2l


def pairwise_distances_logits(a, b):
    n = a.shape[0]
    m = b.shape[0]
    logits = -((a.unsqueeze(1).expand(n, m, -1) -
                b.unsqueeze(0).expand(n, m, -1))**2).sum(dim=2)
    return logits


def conv_block(in_channels, out_channels):
    bn = nn.BatchNorm2d(out_channels)
    nn.init.uniform_(bn.weight)
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        bn,
        nn.ReLU(),
        nn.MaxPool2d(2)
    )


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


class Convnet(nn.Module):

    def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(x_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
        )
        self.out_channels = 1600

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)


def fast_adapt(batch, ways, shot, query_num, metric=pairwise_distances_logits):
    data, _ = [_.cuda() for _ in batch]
    n_items = shot * ways

    embeddings = model(data)
    support, query = embeddings[:n_items], embeddings[n_items:]

    support = support.reshape(shot, ways, -1).mean(dim=0)

    label = torch.arange(ways).repeat(query_num)
    label = label.type(torch.cuda.LongTensor)

    logits = pairwise_distances_logits(query, support)
    loss = F.cross_entropy(logits, label)
    acc = accuracy(logits, label)
    return loss, acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epoch', type=int, default=250)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--test-way', type=int, default=5)
    parser.add_argument('--test-shot', type=int, default=1)
    parser.add_argument('--test-query', type=int, default=30)
    parser.add_argument('--train-query', type=int, default=15)
    parser.add_argument('--train-way', type=int, default=30)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()
    print(args)

    device = torch.device('cpu')
    if args.gpu:
        print("Using gpu")
        torch.cuda.manual_seed(43)
        device = torch.device('cuda')

    model = Convnet()
    model.to(device)

    path_data = '/datadrive/few-shot/miniimagenetdata'
    train_dataset = l2l.vision.datasets.MiniImagenet(
        root=path_data, mode='train')
    valid_dataset = l2l.vision.datasets.MiniImagenet(
        root=path_data, mode='validation')
    test_dataset = l2l.vision.datasets.MiniImagenet(
        root=path_data, mode='test')

    train_sampler = l2l.data.NShotKWayTaskSampler(
        train_dataset.y, 100, args.train_way, args.shot, args.train_query)

    train_loader = DataLoader(dataset=train_dataset, batch_sampler=train_sampler,
                              num_workers=8, pin_memory=True)

    val_sampler = l2l.data.NShotKWayTaskSampler(
        valid_dataset.y, 400, args.test_way, args.shot, args.train_query)

    val_loader = DataLoader(dataset=valid_dataset, batch_sampler=val_sampler,
                            num_workers=8, pin_memory=True)

    test_sampler = l2l.data.NShotKWayTaskSampler(
        test_dataset.y, 2000, args.test_way, args.test_shot, args.test_query)

    test_loader = DataLoader(test_dataset, batch_sampler=test_sampler,
                             num_workers=8, pin_memory=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=20, gamma=0.5)

    for epoch in range(1, args.max_epoch + 1):
        lr_scheduler.step()

        model.train()

        loss_ctr = 0
        n_loss = 0
        n_acc = 0

        for i, batch in enumerate(train_loader, 1):

            loss, acc = fast_adapt(batch, args.train_way, args.shot,
                                   args.train_query, metric=pairwise_distances_logits)

            loss_ctr += 1
            n_loss += loss.item()
            n_acc += acc

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('epoch {}, train, loss={:.4f} acc={:.4f}'.format(
            epoch, n_loss/loss_ctr, n_acc/loss_ctr))

        model.eval()

        loss_ctr = 0
        n_loss = 0
        n_acc = 0
        for i, batch in enumerate(val_loader, 1):
            loss, acc = fast_adapt(batch, args.test_way, args.shot,
                                   args.train_query, metric=pairwise_distances_logits)

            loss_ctr += 1
            n_loss += loss.item()
            n_acc += acc

        print('epoch {}, val, loss={:.4f} acc={:.4f}'.format(
            epoch, n_loss/loss_ctr, n_acc/loss_ctr))

    loss_ctr = 0
    n_acc = 0

    for i, batch in enumerate(test_loader, 1):
        _, acc = fast_adapt(batch, args.test_way, args.test_shot,
                            args.test_query, metric=pairwise_distances_logits)
        loss_ctr += 1
        n_acc += acc
        print('batch {}: {:.2f}({:.2f})'.format(
            i, n_acc/loss_ctr * 100, acc * 100))