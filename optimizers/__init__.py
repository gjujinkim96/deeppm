from utils import recursive_vars

from class_dict_builder import make_class_dict

# class dict build proces
existing_modules = [('torch.optim', 'torch')]
class_dict = make_class_dict(
    existing_modules=existing_modules,
)

def load_optimizer_from_cfg(model, cfg):
    optimizer_setting = recursive_vars(getattr(cfg.train, 'optimizer_setting', {}))
    return load_optimizer(model, cfg.train.optimizer, optimizer_setting)

def load_optimizer(model, optimizer_type, optimizer_setting={}):
    if optimizer_type not in class_dict:
        raise NotImplementedError()
    
    optimizer_class = class_dict[optimizer_type]
    return optimizer_class(model.parameters(), **optimizer_setting)
