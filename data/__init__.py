from utils import recursive_vars
import torch

from .raw_data import RawData
from .ithemal_converter import IthemalConverter
from .string_converter import StringConverter
from .data_holder import DataHolder

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

    return load_data(
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

def load_data_given_paths(cfg, custom_idx_split_path, given_token_mapping_path, small_size=False):
    special_tokens = recursive_vars(cfg.data.special_token_idx)

    data_setting = cfg.data.data_setting

    idx_dict = torch.load(custom_idx_split_path)

    given_token_mapping = torch.load(given_token_mapping_path, map_location=torch.device('cpu'))

    return load_data(
        cfg.data.data_file, 
        small_size=small_size,
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

def load_data(data_savefile, small_size=False, only_unique=False,
    split_mode='none', split_perc=(8, 2, 0),
    special_tokens=None, 
    prepare_mode='stacked', shuffle=False, given_token_mapping=None,
    instr_limit=400, given_train_val_test_idx=None):
    if prepare_mode == 'stacked':
        raw_data = RawData(data_savefile, small_size=small_size, use_metadata=True)
        converter = IthemalConverter(special_tokens=special_tokens, given_token_mapping=given_token_mapping)
    elif prepare_mode == 'stacked_raw':
        raw_data = RawData(data_savefile, small_size=small_size, use_metadata=False)
        converter = StringConverter(special_tokens=special_tokens, given_token_mapping=given_token_mapping)
    else:
        raise NotImplementedError()
    
    data = converter.convert(raw_data, instr_limit=instr_limit)
    data_holder = DataHolder(data, converter)
    
    if only_unique:
        data_holder.cleanse_repeated()

    data_holder.generate_datasets(split_mode=split_mode, split_perc=split_perc,
                                shuffle=shuffle, given_train_val_test_idx=given_train_val_test_idx, small_size=small_size)

    return data_holder

