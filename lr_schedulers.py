import torch.optim as optim

def load_lr_scheduler(optimizer, train_cfg):
    if train_cfg.lr_scheduler == 'LinearLR':
        return optim.lr_scheduler.LinearLR(optimizer)
    else:
        raise NotImplementedError()
    