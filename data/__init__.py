import data.data_cost as dt
from utils import recursive_vars

def load_data_from_cfg(is_small_size, cfg, given_token_mapping=None, given_train_val_test_idx=None):
    special_tokens = getattr(cfg.data, 'special_token_idx', None)
    if special_tokens is not None:
        special_tokens = recursive_vars(special_tokens)

    data_setting = cfg.data.data_setting

    data = dt.load_data(
        cfg.data.data_file, 
        small_size=is_small_size,
        only_unique=data_setting.only_unique,
        split_mode=data_setting.split_mode,
        split_perc=(data_setting.train_perc, data_setting.val_perc, data_setting.test_perc),
        special_tokens=special_tokens,
        prepare_mode=getattr(data_setting, 'prepare_mode', 'stacked'),
        shuffle=getattr(data_setting, 'shuffle', False),
        given_token_mapping=data_setting.given_token_mapping,
        instr_limit=getattr(data_setting, 'instr_limit', 400),
        given_train_val_test_idx=given_train_val_test_idx
    )

    return data
