import torch
import torch.nn as nn
import math

import importlib, inspect


class MapeLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.loss_fn = nn.L1Loss(reduction='none')

    def forward(self, output, target):
        loss = self.loss_fn(output, target) / (target + 1e-5)
        return torch.mean(loss)
       

# https://datascience.stackexchange.com/questions/96271/logcoshloss-on-pytorch
def log_cosh_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    def _log_cosh(x: torch.Tensor) -> torch.Tensor:
        return x + torch.nn.functional.softplus(-2. * x) - math.log(2.0)
    return torch.mean(_log_cosh(y_pred - y_true))

class LogCoshLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, y_pred: torch.Tensor, y_true: torch.Tensor
    ) -> torch.Tensor:
        return log_cosh_loss(y_pred, y_true)


class_dict = {}
module = importlib.import_module('torch.nn', package='torch')
for name, cls in inspect.getmembers(module, inspect.isclass):
    class_dict[name] = cls

module = importlib.import_module('losses')
for name, cls in inspect.getmembers(module, inspect.isclass):
    if cls.__module__ == module.__name__:
        class_dict[name] = cls 


def load_losses(loss_type, loss_setting={}):
    if loss_type not in class_dict:
        raise NotImplementedError()
    
    loss_class = class_dict[loss_type]
    return loss_class(**loss_setting)
