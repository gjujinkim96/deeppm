from utils import recursive_vars
from class_dict_builder import make_class_dict

# class dict build proces
existing_modules = [('torch.nn', 'torch')]
class_dict = make_class_dict(
    'losses',
    custom_use_class=True,
    existing_modules=existing_modules,
)

def load_losses_from_cfg(cfg):
    tmp = getattr(cfg.train, 'loss_setting', {})
    loss_setting = recursive_vars(tmp)
    return load_losses(cfg.train.loss, loss_setting)

def load_losses(loss_type, loss_setting={}):
    if loss_type not in class_dict:
        raise NotImplementedError()
    
    loss_class = class_dict[loss_type]
    return loss_class(**loss_setting)
