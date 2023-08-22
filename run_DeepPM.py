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



def main():
    args = get_args(show=True)
    cfg = get_configs(args, show=True)
    
    set_seeds(cfg.train.seed)
    
    expt = Experiment(args.exp_name)
    if expt.check_root_exist() and not args.exp_override:
        print(f'{expt.experiment_root_path()} exist.')
        return 

    data = load_data_from_cfg(args.small_size, cfg)

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
    

    wandb_log.wandb_init(args,cfg)
    
    dump_obj_to_root(expt, data.dump_dataset_params(), 'data_mapping.dump')
    dump_obj_to_root(expt, cfg, 'config.dump')
    dump_idx_to_root(expt, data)

    trainer = train.Trainer(cfg, model, (train_ds, val_ds, test_ds), expt, 
                            optimizer, lr_scheduler, device, args.small_training)

    # with torch.autograd.detect_anomaly():
    trainer.train()

    wandb_log.wandb_finish()

if __name__ == '__main__':
    main()
