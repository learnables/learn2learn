#!/usr/bin/env python3

import random
import copy
import numpy as np
import torch
import learn2learn as l2l


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


def fast_adapt(batch, learner, adapt_opt, loss, adaptation_steps, shots, ways, batch_size, device):
    data, labels = batch
    data, labels = data.to(device), labels.to(device)

    # Separate data into adaptation/evalutation sets
    adaptation_indices = np.zeros(data.size(0), dtype=bool)
    adaptation_indices[np.arange(shots*ways) * 2] = True
    evaluation_indices = torch.from_numpy(~adaptation_indices)
    adaptation_indices = torch.from_numpy(adaptation_indices)
    adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
    evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]

    # Adapt the model
    for step in range(adaptation_steps):
        idx = torch.randint(
            adaptation_data.size(0),
            size=(batch_size, )
        )
        adapt_X = adaptation_data[idx]
        adapt_y = adaptation_labels[idx]
        adapt_opt.zero_grad()
        error = loss(learner(adapt_X), adapt_y)
        error.backward()
        adapt_opt.step()

    # Evaluate the adapted model
    predictions = learner(evaluation_data)
    valid_error = loss(predictions, evaluation_labels)
    valid_error /= len(evaluation_data)
    valid_accuracy = accuracy(predictions, evaluation_labels)
    return valid_error, valid_accuracy


def main(
        experiment='dev',
        problem='mini-imagenet',
        ways=5,
        train_shots=15,
        test_shots=5,
        meta_lr=1.0,
        meta_bsz=5,
        fast_lr=0.001,
        train_bsz=10,
        test_bsz=15,
        train_steps=8,
        test_steps=50,
        iterations=100000,
        test_interval=100,
        save='',
        cuda=1,
        seed=42,
):
    cuda = bool(cuda)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cpu')
    if cuda and torch.cuda.device_count():
        torch.cuda.manual_seed(seed)
        device = torch.device('cuda')

    train_tasks, valid_tasks, test_tasks = l2l.vision.benchmarks.get_tasksets(
        'mini-imagenet',
        train_samples=2*train_shots,
        train_ways=ways,
        test_samples=2*test_shots,
        test_ways=ways,
        root='~/data',
    )

    # Create model
    model = l2l.vision.models.MiniImagenetCNN(ways)
    model.to(device)
    opt = torch.optim.SGD(model.parameters(), meta_lr)
    adapt_opt = torch.optim.Adam(model.parameters(), lr=fast_lr, betas=(0, 0.999))
    adapt_opt_state = adapt_opt.state_dict()
    loss = torch.nn.CrossEntropyLoss(reduction='mean')

    train_inner_errors = []
    train_inner_accuracies = []
    valid_inner_errors = []
    valid_inner_accuracies = []
    test_inner_errors = []
    test_inner_accuracies = []

    for iteration in range(iterations):
        opt.zero_grad()
        meta_train_error = 0.0
        meta_train_accuracy = 0.0
        meta_valid_error = 0.0
        meta_valid_accuracy = 0.0
        meta_test_error = 0.0
        meta_test_accuracy = 0.0

        # anneal meta-lr
        frac_done = float(iteration) / iterations
        new_lr = frac_done * meta_lr + (1 - frac_done) * meta_lr
        for pg in opt.param_groups:
            pg['lr'] = new_lr

        # zero-grad the parameters
        for p in model.parameters():
            p.grad = torch.zeros_like(p.data)

        for task in range(meta_bsz):
            # Compute meta-training loss
            learner = copy.deepcopy(model)
            adapt_opt = torch.optim.Adam(
                learner.parameters(),
                lr=fast_lr,
                betas=(0, 0.999)
            )
            adapt_opt.load_state_dict(adapt_opt_state)
            batch = train_tasks.sample()
            evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                               learner,
                                                               adapt_opt,
                                                               loss,
                                                               train_steps,
                                                               train_shots,
                                                               ways,
                                                               train_bsz,
                                                               device)
            adapt_opt_state = adapt_opt.state_dict()
            for p, l in zip(model.parameters(), learner.parameters()):
                p.grad.data.add_(-1.0, l.data)

            meta_train_error += evaluation_error.item()
            meta_train_accuracy += evaluation_accuracy.item()

            if iteration % test_interval == 0:
                # Compute meta-validation loss
                learner = copy.deepcopy(model)
                adapt_opt = torch.optim.Adam(
                    learner.parameters(),
                    lr=fast_lr,
                    betas=(0, 0.999)
                )
                adapt_opt.load_state_dict(adapt_opt_state)
                batch = valid_tasks.sample()
                evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                                   learner,
                                                                   adapt_opt,
                                                                   loss,
                                                                   test_steps,
                                                                   test_shots,
                                                                   ways,
                                                                   test_bsz,
                                                                   device)
                meta_valid_error += evaluation_error.item()
                meta_valid_accuracy += evaluation_accuracy.item()

                # Compute meta-testing loss
                learner = copy.deepcopy(model)
                adapt_opt = torch.optim.Adam(
                    learner.parameters(),
                    lr=fast_lr,
                    betas=(0, 0.999)
                )
                adapt_opt.load_state_dict(adapt_opt_state)
                batch = test_tasks.sample()
                evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                                   learner,
                                                                   adapt_opt,
                                                                   loss,
                                                                   test_steps,
                                                                   test_shots,
                                                                   ways,
                                                                   test_bsz,
                                                                   device)
                meta_test_error += evaluation_error.item()
                meta_test_accuracy += evaluation_accuracy.item()

        # Print some metrics
        print('\n')
        print('Iteration', iteration)
        print('Meta Train Error', meta_train_error / meta_bsz)
        print('Meta Train Accuracy', meta_train_accuracy / meta_bsz)
        if iteration % test_interval == 0:
            print('Meta Valid Error', meta_valid_error / meta_bsz)
            print('Meta Valid Accuracy', meta_valid_accuracy / meta_bsz)
            print('Meta Test Error', meta_test_error / meta_bsz)
            print('Meta Test Accuracy', meta_test_accuracy / meta_bsz)

        # Track quantities
        train_inner_errors.append(meta_train_error / meta_bsz)
        train_inner_accuracies.append(meta_train_accuracy / meta_bsz)
        if iteration % test_interval == 0:
            valid_inner_errors.append(meta_valid_error / meta_bsz)
            valid_inner_accuracies.append(meta_valid_accuracy / meta_bsz)
            test_inner_errors.append(meta_test_error / meta_bsz)
            test_inner_accuracies.append(meta_test_accuracy / meta_bsz)

        # Average the accumulated gradients and optimize
        for p in model.parameters():
            p.grad.data.mul_(1.0 / meta_bsz).add_(p.data)
        opt.step()


if __name__ == '__main__':
    main()
