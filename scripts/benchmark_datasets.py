#!/usr/bin/env python3

import time
import torch
import learn2learn as l2l
import torch.multiprocessing as mp
#from multiprocessing import Process, Queue as mQueue
import queue
from tqdm import trange



class ParallelTaskDataset(object):

    def __init__(self, dataset, task_transforms, num_tasks=-1, num_workers=0):
        self.dataset = dataset
        self.task_transforms = task_transforms
        self.num_tasks = num_tasks
        self.num_workers = num_workers
        self.task_dataset = l2l.data.TaskDataset(dataset, task_transforms, num_tasks=num_tasks)
        self.processes = []
        if num_workers > 0:
            self.manager = mp.Manager()
            self.shutdown = self.manager.Value('d', 0)
            self.queue = mp.Queue(num_workers)
            self.processes = [mp.Process(target=ParallelTaskDataset.worker,
                                         args=(dataset, task_transforms, i, self.queue, self.shutdown))
                              for i in range(num_workers)]
            for p in self.processes:
                p.deamon = True
                p.start()

    def __len__(self):
        return len(self.task_dataset)

    def __del__(self):
        try:
            self.shutdown.value = 1
        except:
            pass

    def __getitem__(self, *args, **kwargs):
        return self.task_dataset.__getitem__(*args, **kwargs)

    def sample(self):
        if self.num_workers == 0:
            return self.task_dataset.sample()
        return self.queue.get()
    
    @staticmethod
    def worker(dataset, transforms, i, q, shutdown):
        taskset = l2l.data.TaskDataset(dataset=dataset,
                                       task_transforms=transforms)
        wait_time = 0.1
        try:
            while not shutdown.value:
                task = taskset.sample()
                task[0].share_memory_()
                task[0] = task[0].pin_memory()
                try:
                    q.put(task, timeout=wait_time)
                    wait_time = 0.1
                except queue.Full:
                    wait_time *= 2.0
        except (BrokenPipeError, FileNotFoundError) as e:
            pass




N_LOOP = 32
N_DATA = 5000
SLEEP = 0.15


def work(task_description):
    time.sleep(SLEEP)
    return task_description

if __name__ == '__main__':
    mp.set_start_method('spawn')

    X, y = torch.randn(N_DATA, 10000), torch.randint(1000, size=(N_DATA, 1))
    dataset = torch.utils.data.TensorDataset(X, y)
    dataset = l2l.vision.datasets.MiniImagenet(root='~/data')
    dataset = l2l.data.MetaDataset(dataset)
    transforms = [l2l.data.transforms.FilterLabels(dataset, list(range(230))),
#                  work,
#                  l2l.data.transforms.KShots(dataset, k=1),
#                  l2l.data.transforms.NWays(dataset, n=5),
                  l2l.data.transforms.FusedNWaysKShots(dataset, n=5, k=1),
                  l2l.data.transforms.LoadData(dataset),
                  l2l.data.transforms.RemapLabels(dataset),
                  l2l.data.transforms.ConsecutiveLabels(dataset),
    ]

    sequential_tasks = l2l.data.TaskDataset(dataset,
                                            task_transforms=transforms)
    parallel_tasks = ParallelTaskDataset(dataset,
                                         task_transforms=transforms,
                                         num_workers=4)

    for _ in range(10):
        asdf = torch.randn(20).cuda()
    start = time.time()
    for t in trange(N_LOOP):
        data = parallel_tasks.sample()
        data[0].pin_memory()
        asdf = data[0].cuda()
        time.sleep(SLEEP)
    end = time.time()
    print('Parallel:', end - start)
    #del parallel_tasks

    start = time.time()
    for t in trange(N_LOOP):
        data = sequential_tasks.sample()
        data[0].pin_memory()
        asdf = data[0].cuda()
        time.sleep(SLEEP)
    end = time.time()
    print('Sequential:', end - start)

