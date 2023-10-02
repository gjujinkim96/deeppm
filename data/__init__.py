import data.data_cost as dt
from utils import recursive_vars
import torch

def load_data_from_cfg(is_small_size, cfg):
    special_tokens = recursive_vars(cfg.data.special_token_idx)

    data_setting = cfg.data.data_setting

    if data_setting.custom_idx_split is not None:
        idx_dict = torch.load(data_setting.custom_idx_split)
    else:
        idx_dict = None

    if data_setting.given_token_mapping is not None:
        given_token_mapping = torch.load(data_setting.given_token_mapping, map_location=torch.device('cpu'))[0]
    else:
        given_token_mapping = None

    data = dt.load_data(
        cfg.data.data_file, 
        small_size=is_small_size,
        only_unique=data_setting.only_unique,
        split_mode=data_setting.split_mode,
        split_perc=(data_setting.train_perc, data_setting.val_perc, data_setting.test_perc),
        special_tokens=special_tokens,
        prepare_mode=getattr(data_setting, 'prepare_mode', 'stacked'),
        shuffle=getattr(data_setting, 'shuffle', False),
        given_token_mapping=given_token_mapping,
        instr_limit=getattr(data_setting, 'instr_limit', 400),
        given_train_val_test_idx=idx_dict
    )

    return data
