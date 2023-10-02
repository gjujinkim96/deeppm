from utils import recursive_vars
from class_dict_builder import make_class_dict

# class dict build process
existing_modules = [('torch.optim.lr_scheduler', 'torch')]
class_dict = make_class_dict(
    'lr_schedulers',
    custom_use_class=True, 
    custom_use_function=True, 
    existing_modules=existing_modules
)

#  loading part
def load_lr_scheduler_from_cfg(optimizer, cfg):
    lr_scheduler_setting = recursive_vars(getattr(cfg.train, 'lr_scheduler_setting', {}))
    return load_lr_scheduler(optimizer, cfg.train.lr_scheduler, lr_scheduler_setting)

def load_batch_lr_scheduler_from_cfg(optimizer, cfg, train_ds):
    training_step = int((len(train_ds) + cfg.train.batch_size - 1) / cfg.train.batch_size)
    total_step = training_step * cfg.train.n_epochs
    lr_sched_setting = recursive_vars(cfg.train.lr_scheduler_setting)
    lr_sched_setting['total_step'] = total_step
    
    return load_lr_scheduler(optimizer, cfg.train.lr_scheduler, lr_sched_setting)

def load_lr_scheduler(optimizer, lr_scheduler_type, lr_scheduler_setting={}):
    if lr_scheduler_type not in class_dict:
        raise NotImplementedError()
    
    optimizer_class = class_dict[lr_scheduler_type]
    return optimizer_class(optimizer, **lr_scheduler_setting)


