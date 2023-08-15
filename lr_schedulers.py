import torch.optim.lr_scheduler as lr_sched
import importlib, inspect
from utils import recursive_vars

# elif train_cfg.lr_scheduler == 'StepLR':
#     return optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1/1.2)
# elif train_cfg.lr_scheduler == 'DecayAfterDelay':
#     return optim.lr_scheduler.LambdaLR(optimizer, decay_after_2_delay)

# def load_batch_step_lr_scheduler(optimizer, train_cfg, train_step):
#     if train_cfg.lr_scheduler == 'OneThirdLR':
#         end = int(train_step * 0.3)
#         return optim.lr_scheduler.MultiStepLR(optimizer, milestones=[end], gamma=0.1)
#     # elif train_cfg.lr_scheduler == 'CyclicLR':
#     #     return optim.lr_scheduler.CyclicLR(optimizer, train_cfg.lr, train_cfg.lr * 2)
#     else:
#         raise NotImplementedError()
    
# def decay_after_delay(epoch):
#     if epoch <= 2:
#         return 1
#     return 1/1.2**(epoch-2)

# # epoch starts from 0
# def decay_after_2_delay(epoch):
#     if epoch <= 1:
#         return 1
#     return 1/1.2**(epoch-1)


def warmup_with_decay(warmup_step, total_step):
    def lr_sched(step):
        step += 1
        if step <= warmup_step:
            return step / warmup_step
        else:
            return (total_step - step) / (total_step - warmup_step)
    return lr_sched

def get_warmp_with_decay_lr_sched(opt, total_step, warmup_step=None, warmup_ratio=0.1):
    if warmup_step is None:
        warmup_step = int(warmup_ratio * total_step)
    return lr_sched.LambdaLR(opt, warmup_with_decay(warmup_step=warmup_step, total_step=total_step))


class_dict = {}
module = importlib.import_module('torch.optim.lr_scheduler', package='torch')
for name, cls in inspect.getmembers(module, inspect.isclass):
    class_dict[name] = cls

module = importlib.import_module('lr_schedulers')
for name, cls in inspect.getmembers(module, inspect.isclass):
    if cls.__module__ == module.__name__:
        class_dict[name] = cls 

for name, cls in inspect.getmembers(module, inspect.isfunction):
    if cls.__module__ == module.__name__:
        class_dict[name] = cls 

def load_lr_scheduler_from_cfg(optimizer, cfg):
    return load_lr_scheduler(optimizer, cfg.train.lr_scheduler, recursive_vars(cfg.train.lr_scheduler_setting))

def load_batch_lr_scheduler_from_cfg(optimizer, cfg, total_step):
    lr_sched_setting = recursive_vars(cfg.train.lr_scheduler_setting)
    lr_sched_setting['total_step'] = total_step
    
    return load_lr_scheduler(optimizer, cfg.train.lr_scheduler, lr_sched_setting)


def load_lr_scheduler(optimizer, lr_scheduler_type, lr_scheduler_setting={}):
    if lr_scheduler_type not in class_dict:
        raise NotImplementedError()
    
    optimizer_class = class_dict[lr_scheduler_type]
    return optimizer_class(optimizer, **lr_scheduler_setting)

