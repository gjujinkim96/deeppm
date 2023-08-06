import torch.optim as optim

def load_lr_scheduler(optimizer, train_cfg):
    if train_cfg.lr_scheduler == 'LinearLR':
        return optim.lr_scheduler.LinearLR(optimizer, total_iters=train_cfg.lr_total_iters)
    elif train_cfg.lr_scheduler == 'StepLR':
        return optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1/1.2)
    elif train_cfg.lr_scheduler == 'DecayAfterDelay':
        return optim.lr_scheduler.LambdaLR(optimizer, decay_after_2_delay)
    else:
        raise NotImplementedError()
    
def load_batch_step_lr_scheduler(optimizer, train_cfg, train_step):
    if train_cfg.lr_scheduler == 'OneThirdLR':
        end = int(train_step * 0.3)
        return optim.lr_scheduler.MultiStepLR(optimizer, milestones=[end], gamma=0.1)
    # elif train_cfg.lr_scheduler == 'CyclicLR':
    #     return optim.lr_scheduler.CyclicLR(optimizer, train_cfg.lr, train_cfg.lr * 2)
    else:
        raise NotImplementedError()
    
def decay_after_delay(epoch):
    if epoch <= 2:
        return 1
    return 1/1.2**(epoch-2)

# epoch starts from 0
def decay_after_2_delay(epoch):
    if epoch <= 1:
        return 1
    return 1/1.2**(epoch-1)


