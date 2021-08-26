#!/usr/bin/env python

import os
import wandb
import random
import numpy as np
import torch
import tqdm
import dataclasses
import simple_parsing as sp
import learn2learn as l2l
import torchvision as tv
import kornia


def fast_adapt(
    task,
    features,
    classifier,
    shots,
    ways,
    device,
):
    data, labels = task
    data = features(data.to(device))
    data = (data - data.mean(dim=0, keepdim=True))
    labels = labels.long().to(device)

    (support_data, support_labels), (query_data, query_labels) = l2l.data.partition_task(
        data=data,
        labels=labels,
        shots=args.options.shots,
        ways=args.options.ways,
    )
    classifier.fit_(support_data, support_labels)
    preds = classifier(query_data)
    acc = l2l.utils.accuracy(preds, query_labels)
    loss = torch.nn.functional.cross_entropy(preds, query_labels)
    return acc, loss


def pretrain(args):

    # Setup torch
    device = torch.device('cpu')
    random.seed(args.options.seed)
    np.random.seed(args.options.seed)
    torch.manual_seed(args.options.seed)
    if args.options.cuda and torch.cuda.device_count():
        device = torch.device('cuda')
        torch.cuda.manual_seed(args.options.seed)

    # Setup wandb
    wandb.init(
        project='meta-features',
        name=f'{args.options.dataset}-{args.options.model}',
        config=l2l.utils.flatten_config(args),
        mode='online' if args.options.use_wandb else 'disabled',
    )

    # Setup dataset
    data_device = device
    if args.options.dataset == 'tiered-imagenet':
        data_device = torch.device('cpu')
    train_tasks, valid_tasks, test_tasks = l2l.vision.benchmarks.get_tasksets(
        name=args.options.dataset,
        train_ways=args.options.ways,
        train_samples=2*args.options.shots,
        test_ways=args.options.ways,
        test_samples=2*args.options.shots,
        device=data_device,
    )

    train_dataset = train_tasks.dataset
    num_classes = int(max(train_dataset.labels)) + 1
    train_augmentation = []
    if args.options.data_augmentation == 'jitter':  # for MI / TI
        train_augmentation = [
            kornia.augmentation.RandomHorizontalFlip(),
            kornia.augmentation.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        ]
    elif args.options.data_augmentation == 'cut':  # for CFS / FC100
        train_augmentation = [
            kornia.augmentation.RandomHorizontalFlip(),
            kornia.augmentation.RandomErasing(),
        ]
    train_augmentation = tv.transforms.Compose(train_augmentation)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.options.batch_size,
        shuffle=True,
    )
    train_loader = l2l.data.InfiniteIterator(train_loader)

    # Setup model
    if args.options.model == 'cnn4':
        if args.options.dataset == 'cifar-fs' \
                or args.options.dataset == 'fc100':
            model = l2l.vision.models.CNN4(
                output_size=num_classes,
                hidden_size=64,
                embedding_size=256,
            )
    elif args.options.model == 'resnet12':
        if args.options.dataset == 'mini-imagenet' \
                or args.options.dataset == 'tiered-imagenet':
            model = l2l.vision.models.ResNet12(num_classes)
        elif args.options.dataset == 'cifar-fs' \
                or args.options.dataset == 'fc100':
            model = l2l.vision.models.ResNet12(num_classes, dropblock_size=2)
    if not args.options.load_weights == '':
        model.features.load_state_dict(torch.load(args.options.load_weights))
    model.to(device)
    features = model.features
    protonet = l2l.nn.PrototypicalClassifier(distance='cosine')

    # Setup optimization
    opt = torch.optim.SGD(
        params=model.parameters(),
        lr=args.options.lr,
        momentum=0.9,
        nesterov=True,
        weight_decay=args.options.weight_decay,
    )
    schedule = torch.optim.lr_scheduler.MultiStepLR(
        optimizer=opt,
        milestones=[int(0.33 * args.options.iterations), int(0.66 * args.options.iterations)],
        gamma=0.1,
    )
    criterion = torch.nn.CrossEntropyLoss()

    # Training loop
    for iteration in tqdm.trange(
        args.options.iterations,
        desc='Training',
        leave=False,
    ):

        # Training step
        model.train()
        X, y = next(train_loader)
        X = X.to(device)
        X = train_augmentation(X)
        y = y.long().to(device)
        preds = model(X)
        loss = criterion(preds, y)
        acc = l2l.utils.accuracy(preds, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        schedule.step()
        wandb.log({
            'supervised/cross-entropy': loss.item(),
            'supervised/accuracy': acc.item(),
            'iteration': iteration
        }, step=iteration)

        # Validation with protonet
        if iteration % args.options.eval_freq == 0:
            model.eval()
            with torch.no_grad():
                train_acc, train_loss = fast_adapt(
                    task=train_tasks.sample(),
                    features=features,
                    classifier=protonet,
                    shots=args.options.shots,
                    ways=args.options.ways,
                    device=device,
                )
                valid_acc, valid_loss = fast_adapt(
                    task=valid_tasks.sample(),
                    features=features,
                    classifier=protonet,
                    shots=args.options.shots,
                    ways=args.options.ways,
                    device=device,
                )
                test_acc, test_loss = fast_adapt(
                    task=test_tasks.sample(),
                    features=features,
                    classifier=protonet,
                    shots=args.options.shots,
                    ways=args.options.ways,
                    device=device,
                )
                wandb.log({
                    'train/accuracy': train_acc.item(),
                    'train/cross-entropy': train_loss.item(),
                    'valid/accuracy': valid_acc.item(),
                    'valid/cross-entropy': valid_loss.item(),
                    'test/accuracy': test_acc.item(),
                    'test/cross-entropy': test_loss.item(),
                    'iteration': iteration,
                }, step=iteration)

    # Benchmark on 2k tasks
    model.eval()
    num_eval_tasks = 2000
    train_accuracy = 0.0
    valid_accuracy = 0.0
    test_accuracy = 0.0
    with torch.no_grad():
        for _ in tqdm.trange(num_eval_tasks, desc='Evaluation', leave=False):
            train_acc, train_loss = fast_adapt(
                task=train_tasks.sample(),
                features=features,
                classifier=protonet,
                shots=args.options.shots,
                ways=args.options.ways,
                device=device,
            )
            valid_acc, valid_loss = fast_adapt(
                task=valid_tasks.sample(),
                features=features,
                classifier=protonet,
                shots=args.options.shots,
                ways=args.options.ways,
                device=device,
            )
            test_acc, test_loss = fast_adapt(
                task=test_tasks.sample(),
                features=features,
                classifier=protonet,
                shots=args.options.shots,
                ways=args.options.ways,
                device=device,
            )
            train_accuracy += train_acc
            valid_accuracy += valid_acc
            test_accuracy += test_acc
    train_accuracy = train_accuracy/num_eval_tasks*100.
    valid_accuracy = valid_accuracy/num_eval_tasks*100.
    test_accuracy = test_accuracy/num_eval_tasks*100.
    print(f'Train Acc: {train_accuracy:,.2f}')
    print(f'Valid Acc: {valid_accuracy:,.2f}')
    print(f'Test Acc: {test_accuracy:,.2f}')
    wandb.log({
        'train/final-accuracy': train_accuracy,
        'valid/final-accuracy': valid_accuracy,
        'test/final-accuracy': test_accuracy,
    })

    # Save weights to disk
    if args.options.save_weights:
        weights_path = os.path.join(
            'saved_models',
            'supervised_pretraining',
            args.options.dataset,
            args.options.model,
        )
        os.makedirs(weights_path, exist_ok=True)
        wandb_id = wandb.run.id
        weights_path = os.path.join(weights_path, f'{wandb_id}.pth')
        torch.save(features.state_dict(), weights_path)


if __name__ == "__main__":

    @dataclasses.dataclass
    class PretrainArgs:

        model: str = 'resnet12'
        dataset: str = 'mini-imagenet'
        iterations: int = 150000
        ways: int = 5
        shots: int = 5
        lr: float = 0.01
        weight_decay: float = 0.0005
        data_augmentation: str = 'jitter'
        batch_size: int = 32
        eval_freq: int = 10
        cuda: bool = True
        seed: int = 1234
        use_wandb: bool = True
        save_weights: bool = False
        load_weights: str = ''

    parser = sp.ArgumentParser(add_dest_to_option_strings=True)
    parser.add_arguments(PretrainArgs, dest='options')
    args = parser.parse_args()
    pretrain(args)
