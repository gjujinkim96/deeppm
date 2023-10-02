import torch
import torch.nn as nn

class MapeLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.loss_fn = nn.L1Loss(reduction='none')

    def forward(self, output, target):
        loss = self.loss_fn(output, target) / (target + 1e-5)
        return torch.mean(loss)
