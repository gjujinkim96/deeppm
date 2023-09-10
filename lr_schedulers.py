import torch.optim.lr_scheduler as lr_sched
import importlib, inspect
from utils import recursive_vars

def decay_after_2_delay(epoch):
    if epoch <= 1:
        return 1
    return 1/1.2**(epoch-1)


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


def decay_after_delay(delay=2, div_factor=1.2):
    def lr_sched(step):
        if step < delay:
            return 1
        return 1/1.2**(step-delay+1)
    return lr_sched

def get_decay_after_delay_lr_sched(opt, delay=2, div_factor=1.2):
    return lr_sched.LambdaLR(opt, decay_after_delay(delay=delay, div_factor=div_factor))

# class dict build process
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

#  loading part
def load_lr_scheduler_from_cfg(optimizer, cfg):
    lr_scheduler_setting = recursive_vars(getattr(cfg.train, 'lr_scheduler_setting', {}))
    return load_lr_scheduler(optimizer, cfg.train.lr_scheduler, lr_scheduler_setting)

def load_batch_lr_scheduler_from_cfg(optimizer, cfg, total_step):
    lr_sched_setting = recursive_vars(cfg.train.lr_scheduler_setting)
    lr_sched_setting['total_step'] = total_step
    
    return load_lr_scheduler(optimizer, cfg.train.lr_scheduler, lr_sched_setting)


def load_lr_scheduler(optimizer, lr_scheduler_type, lr_scheduler_setting={}):
    if lr_scheduler_type not in class_dict:
        raise NotImplementedError()
    
    optimizer_class = class_dict[lr_scheduler_type]
    return optimizer_class(optimizer, **lr_scheduler_setting)

