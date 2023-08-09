import importlib, inspect
from utils import recursive_vars

    
class_dict = {}
module = importlib.import_module('torch.optim', package='torch')
for name, cls in inspect.getmembers(module, inspect.isclass):
    class_dict[name] = cls


def load_optimizer_from_cfg(model, cfg):
    return load_optimizer(model, cfg.train.optimizer, recursive_vars(cfg.train.optimizer_setting))

def load_optimizer(model, optimizer_type, optimizer_setting={}):
    if optimizer_type not in class_dict:
        raise NotImplementedError()
    
    optimizer_class = class_dict[optimizer_type]
    return optimizer_class(model.parameters(), **optimizer_setting)