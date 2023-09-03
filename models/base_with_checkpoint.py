import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

def method_dummy_wrapper(func):
    def func_with_dummy(dummy, x, *args):
        return func(x, *args)
    return func_with_dummy
