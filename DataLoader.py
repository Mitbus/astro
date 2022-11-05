from multiprocessing import Queue, Process, freeze_support
import time
import os
import torch
import numpy as np
from Config import *

eof = None

def read_images(begin, end):
    for i in range(begin, end):
        wait = False
        while not (os.path.isfile(f'train/x_{i}.npy') and os.path.isfile(f'train/y_{i}.npy')):
            if not wait:
                print(f'File y_{i}.npy or x_{i}.npy not exists. Waiting...')
                wait = True
            time.sleep(5)
        if wait:
            print('Reading data:', i)
        xs = np.load(f'train/x_{i}.npy')
        ys = np.load(f'train/y_{i}.npy')
        yield xs, ys

def _async_queue_manager(gen_func, args, max_qsize, queue: Queue):
    for item in gen_func(*args):
        queue.put(item)
        while queue.qsize() >= max_qsize:
            time.sleep(5)
    queue.put(eof)

def iter_asynchronously(gen_func, args, max_qsize):
    q = Queue()
    p = Process(target=_async_queue_manager, args=(gen_func, args, max_qsize, q))
    p.start()
    while True:
        item = q.get()
        if item is eof:
            break
        else:
            yield item

def data_loader(begin, end):
    epoch = 0
    total_step = 0
    while True:
        step = 0
        print(f'Epoch: {epoch}     ')
        time.sleep(5)
        for xs, ys in read_images(begin, end):
            for j in range(0, xs.shape[0], batch_size):
                x = torch.tensor(xs[j:j+batch_size], device=device, dtype=torch.float32)
                y = torch.tensor(ys[j:j+batch_size], device=device, dtype=torch.float32)
                # noise = torch.normal(mean=0, std=0.5, size=x.shape, device=device, dtype=torch.float32)
                # noise2 = (torch.rand(size=x.shape, device=device, dtype=torch.float32) < 0.9).float()
                yield x, y, step, total_step, epoch
                step += 1
                total_step += 1
        epoch += 1