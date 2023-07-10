import torch
import torch.nn as nn

def mse_loss(output, target):

    loss_fn = nn.MSELoss(reduction= None)
    loss = torch.sqrt(loss_fn(output, target)) / (target + 1e-3)
    loss = torch.mean(loss)

    return loss
