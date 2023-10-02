from utils import recursive_vars
from class_dict_builder import make_class_dict

# class dict build proces
class_dict = make_class_dict(
    'datasets',
    custom_use_class=True
)


def load_dataset_from_cfg(data, cfg, show=False):
    train_dataset, val_dataset, test_dataset = load_dataset(data, cfg.data.dataset_class, recursive_vars(cfg.data.dataset_setting),
                        recursive_vars(cfg.data.special_token_idx))
    if show:
        print(f'Train Dataset: {len(train_dataset)}  Val Dataset: {len(val_dataset)}  Test Dataset: {len(test_dataset)}')
    return train_dataset, val_dataset, test_dataset
    

def load_dataset(data, dataset_type, dataset_setting={}, special_tokens={}):
    if dataset_type not in class_dict:
        raise NotImplementedError()
    
    dataset_class = class_dict[dataset_type]
    dataset_setting['is_training'] = True
    train_dataset = dataset_class(data.train, special_tokens=special_tokens, **dataset_setting)

    dataset_setting['is_training'] = False
    val_dataset = dataset_class(data.val, special_tokens=special_tokens, **dataset_setting)
    test_dataset = dataset_class(data.test, special_tokens=special_tokens, **dataset_setting)

    return train_dataset, val_dataset, test_dataset

