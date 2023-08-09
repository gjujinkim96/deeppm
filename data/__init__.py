import data.data_cost as dt
from utils import recursive_vars

def load_data_from_cfg(args, cfg):
    special_tokens = getattr(cfg.data, 'special_token_idx', None)
    if special_tokens is not None:
        special_tokens = recursive_vars(special_tokens)
    data = dt.load_data(
        cfg.data.data_file, 
        small_size=args.small_size,
        stacked=cfg.data.data_setting.stacked, 
        only_unique=cfg.data.data_setting.only_unique,
        hyperparameter_test=cfg.train.hyperparameter_testing.using,
        hyperparameter_test_mult=cfg.train.hyperparameter_testing.mult,
        short_only=cfg.data.data_setting.short_only,
        special_tokens=special_tokens
    )

    return data
