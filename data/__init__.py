import data.data_cost as dt
from utils import recursive_vars

def load_data_from_cfg(args, cfg):
    special_tokens = getattr(cfg.data, 'special_token_idx', None)
    if special_tokens is not None:
        special_tokens = recursive_vars(special_tokens)

    data_setting = cfg.data.data_setting

    data = dt.load_data(
        cfg.data.data_file, 
        small_size=args.small_size,
        stacked=data_setting.stacked, 
        only_unique=data_setting.only_unique,
        split_mode=data_setting.split_mode,
        split_perc=(data_setting.train_perc, data_setting.val_perc, data_setting.test_perc),
        hyperparameter_test=cfg.train.hyperparameter_testing.using,
        hyperparameter_test_mult=cfg.train.hyperparameter_testing.mult,
        special_tokens=special_tokens,
        simplify=getattr(data_setting, 'simplify', False),
        src_info_file=getattr(cfg.data, 'src_info_file', None),
        bert=data_setting.bert
    )

    return data
