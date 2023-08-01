import torch.optim as optim

def load_lr_scheduler(optimizer, train_cfg):
    if train_cfg.lr_scheduler == 'LinearLR':
        return optim.lr_scheduler.LinearLR(optimizer, total_iters=train_cfg.lr_total_iters)
    elif train_cfg.lr_scheduler == 'StepLR':
        return optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1/1.2)
    else:
        raise NotImplementedError()
    