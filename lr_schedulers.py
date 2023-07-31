import torch.optim as optim

def load_lr_scheduler(optimizer, train_cfg):
    if train_cfg.lr_scheduler == 'LinearLR':
        return optim.lr_scheduler.LinearLR(optimizer, total_iters=train_cfg.lr_total_iters)
    else:
        raise NotImplementedError()
    