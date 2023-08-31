from handle_inputs import get_args, get_configs, dict_to_simple_namespace
from data import load_data_from_cfg
from collections import namedtuple
import models
import train
import wandb_log

from utils import set_seeds, get_device, recursive_vars
from DeepPM_utils import *
from experiments.experiment import Experiment
import losses 

import dataset as ds
import data.data_cost as dt
import optimizers as opt
import lr_schedulers as lr_sch
from pathlib import Path



def main():
    args = get_args(show=True)
    cfg = get_configs(args, show=True)
    
    set_seeds(cfg.train.seed)
    
    expt = Experiment(args.exp_name)
    if expt.check_root_exist() and not args.exp_override:
        print(f'{expt.experiment_root_path()} exist.')
        return 
    
    dm = idx_dict = None
    if getattr(cfg, 'pretrained', None) is not None:
        model_root = Path(cfg.pretrained.saved_path)

        config_file = model_root.joinpath('config.dump')
        if not config_file.is_file():
            raise ValueError('No config file is found')

        dm_file = model_root.joinpath('data_mapping.dump')
        if not dm_file.is_file():
            print('No data_mapping.dump is found. Using None instead.')
            dm = None
        else:
            dm = torch.load(dm_file)[0]

        idx_dict_file = model_root.joinpath('idx_dict.dump')
        if not idx_dict_file.is_file():
            print('No idx_dict.dump is found. Using None instead.')
            idx_dict = None
        else:
            idx_dict = torch.load(idx_dict_file)

        pretrained_cfg = torch.load(config_file)
        cfg.data = pretrained_cfg.data

        pretrained_file = model_root.joinpath(cfg.pretrained.using_model_file)
        model_dump = torch.load(pretrained_file, map_location='cpu')
        pretrained = models.load_model_from_cfg(pretrained_cfg)
        pretrained.load_state_dict(model_dump['model'])

        for param in pretrained.parameters():
            param.requires_grad = False

        cfg.pretrained.model = None

    data = load_data_from_cfg(args.small_size, cfg, dm, idx_dict)

    special_tokens = ['PAD', 'SRCS', 'DSTS', 'UNK', 'END', 'MEM', "MEM_FIN", "START", "OP", "INS_START", "INS_END"]
    if getattr(cfg.data, 'special_token_idx', None) is None:
        special_token_idx = {}
    else:
        special_token_idx = recursive_vars(cfg.data.special_token_idx)

    for raw_token in special_tokens:
        if raw_token not in special_token_idx:
            splitted = raw_token.split('_')
            if len(splitted) > 1 and splitted[1] == 'FIN':
                token = f'</{raw_token}>'
            else:
                token = f'<{raw_token}>'

            special_token_idx[raw_token] = data.hot_idxify(token)
    
    cfg.data.special_token_idx = dict_to_simple_namespace(special_token_idx)

    if getattr(cfg.model.model_setting, 'vocab_size', None) is not None:
        if cfg.model.model_setting.vocab_size == -1:
            cfg.model.model_setting.vocab_size = len(data.token_to_hot_idx)

    if getattr(cfg.data.dataset_setting, 'vocab_size', None) is not None:
        if cfg.data.dataset_setting.vocab_size == -1:
            cfg.data.dataset_setting.vocab_size = len(data.token_to_hot_idx)
    
    device = get_device()
    train_ds, val_ds, test_ds = ds.load_dataset_from_cfg(data, cfg, show=True)

    model = models.load_model_from_cfg(cfg)

    optimizer = opt.load_optimizer_from_cfg(model, cfg)

    
    if cfg.train.use_batch_step_lr:
        training_step = int((len(train_ds) + cfg.train.batch_size - 1) / cfg.train.batch_size)
        total_step = training_step * cfg.train.n_epochs
        lr_scheduler = lr_sch.load_batch_lr_scheduler_from_cfg(optimizer, cfg, total_step)
    else:
        lr_scheduler = lr_sch.load_lr_scheduler_from_cfg(optimizer, cfg)
    

    wandb_log.wandb_init(args, cfg)

    if getattr(cfg, 'pretrained', None) is not None:
        cfg.pretrained.model = pretrained
    
    dump_obj_to_root(expt, data.dump_dataset_params(), 'data_mapping.dump')
    dump_obj_to_root(expt, cfg, 'config.dump')
    dump_idx_to_root(expt, data)

    is_bert = getattr(cfg.train, 'is_bert', False)
    trainer = train.Trainer(cfg, model, (train_ds, val_ds, test_ds), expt, 
                            optimizer, lr_scheduler, device, args.small_training, is_bert=is_bert)

    # with torch.autograd.detect_anomaly():
    trainer.train()

    wandb_log.wandb_finish()

if __name__ == '__main__':
    main()
