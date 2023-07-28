import torch
import torch.nn as nn
import math

def load_loss_fn(train_cfg):
    if train_cfg.loss_fn == 'mape':
        return mape_loss
    else:
        raise NotImplementedError()
    
def mape_loss(output, target):

    # loss_fn = nn.MSELoss(reduction='none')
    # loss = torch.sqrt(loss_fn(output, target) + 1e-5) / (target + 1e-5)
    # loss = torch.mean(loss)

    loss_fn = nn.L1Loss(reduction='none')
    loss = loss_fn(output, target) / (target + 1e-5)
    loss = torch.mean(loss)

    return loss

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
    