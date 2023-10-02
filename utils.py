""" Utils Functions """
import random

import numpy as np
import torch

from types import SimpleNamespace



def recursive_vars(x):
    if not isinstance(x, dict):
        x = vars(x)
    ret = {}
    for k, v in x.items():
        if isinstance(v, SimpleNamespace):
            v = recursive_vars(v)
        ret[k] = v
    return ret

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_worker_generator(seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def set_seeds(seed):
    "set random seeds"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device(should_print=True):
    if torch.cuda.is_available():
        str_device = 'cuda'
    else:
        str_device = 'cpu'
        
    device = torch.device(str_device)

    if should_print:
        print(f'Using {device}')
    return device

def correct_regression(pred, answer, tolerance):
    if isinstance(pred, list):
        pred = torch.tensor(pred)
    if isinstance(answer, list):
        answer = torch.tensor(answer)

    percentage = abs(pred - answer) * 100.0 / (answer + 1e-3)
    return sum(percentage < tolerance)
