import torch
import torch.nn as nn
from utils import recursive_vars

import importlib, inspect


class MapeLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.loss_fn = nn.L1Loss(reduction='none')

    def forward(self, output, target):
        loss = self.loss_fn(output, target) / (target + 1e-5)
        return torch.mean(loss)
    
class_dict = {}
module = importlib.import_module('torch.nn', package='torch')
for name, cls in inspect.getmembers(module, inspect.isclass):
    class_dict[name] = cls

module = importlib.import_module('losses')
for name, cls in inspect.getmembers(module, inspect.isclass):
    if cls.__module__ == module.__name__:
        class_dict[name] = cls 

def load_losses_from_cfg(cfg):
    tmp = getattr(cfg.train, 'loss_setting', {})
    loss_setting = recursive_vars(tmp)
    return load_losses(cfg.train.loss, loss_setting)

def load_losses(loss_type, loss_setting={}):
    if loss_type not in class_dict:
        raise NotImplementedError()
    
    loss_class = class_dict[loss_type]
    return loss_class(**loss_setting)
