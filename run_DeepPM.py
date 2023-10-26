from handle_inputs import get_args, get_configs
from data import load_data_from_cfg
import models
import train
import wandb_log

from utils import set_seeds, get_device
from experiment import Experiment, KFoldExperiments

import datasets as ds
import optimizers as opt
import lr_schedulers as lr_sch
import losses as ls

from dumper import Dumper
import numpy as np
from sklearn.model_selection import StratifiedKFold
from data.data_holder import get_group

def kfold(args, cfg):
    print("[jsshim-debug] print args")
    print(args)

    device = get_device()
    print(device)

    data_holder = load_data_from_cfg(args.small_size, cfg)

    #exit()
    
    # kfold stuff
    using_code_ids = set()
    for datum in data_holder.data.train:
        using_code_ids.add(datum.code_id)

    for datum in data_holder.data.val:
        using_code_ids.add(datum.code_id)

    code_id_mapping = {}
    for i, datum in enumerate(data_holder.data):
        if datum.code_id in using_code_ids:
            code_id_mapping[datum.code_id] = i
    
    x = np.array(list(using_code_ids))
    y = []
    for code_id in using_code_ids:
        data_idx = code_id_mapping[code_id]
        datum = data_holder.data[data_idx]
        group = get_group(datum.block.num_instrs())
        y.append(group)
    y = np.array(y)


    k = cfg.train.kfold.k
    kfold_expt = KFoldExperiments(args.exp_name, exp_existing=args.exp_override, k=k)
    kfold_expt.restart()
    skf = StratifiedKFold(n_splits=k)
    exp_name = args.exp_name

    for i, (train_idx, val_idx) in enumerate(skf.split(x, y)):
        print(f'##### Fold {i} #####')
        train_code_id = x[train_idx]
        val_code_id = x[val_idx]

        data_holder.mix_train_val(train_code_id, val_code_id, code_id_mapping)

        model = models.load_model_from_cfg(cfg)
        loss_fn = ls.load_losses_from_cfg(cfg)
        optimizer = opt.load_optimizer_from_cfg(model, cfg)

        expt = kfold_expt.folds[i]
        dumper = Dumper(expt)
        args.exp_name = f'{exp_name}_{i}'

        train_ds, val_ds, _ = ds.load_dataset_from_cfg(data_holder, cfg, show=True)
    
        if cfg.train.use_batch_step_lr:
            lr_scheduler = lr_sch.load_batch_lr_scheduler_from_cfg(optimizer, cfg, train_ds)
        else:
            lr_scheduler = lr_sch.load_lr_scheduler_from_cfg(optimizer, cfg)
    
        wandb_log.wandb_init(args, cfg, group=exp_name)
    
        dumper.dump_data_holder(data_holder)
        dumper.dump_config(cfg)

        trainer = train.Trainer(cfg, model, (train_ds, val_ds), dumper, 
                                optimizer, lr_scheduler, loss_fn, device, args.small_training)

    # with torch.autograd.detect_anomaly():
        trainer.train()

        wandb_log.wandb_finish()
        print()

def normal(args, cfg):
    set_seeds(cfg.train.seed)
    
    expt = Experiment(args.exp_name, exp_existing=args.exp_override)
    expt.restart()
    dumper = Dumper(expt)

    device = get_device()

    data_holder = load_data_from_cfg(args.small_size, cfg)

    train_ds, val_ds, _ = ds.load_dataset_from_cfg(data_holder, cfg, show=True)
    model = models.load_model_from_cfg(cfg)
    loss_fn = ls.load_losses_from_cfg(cfg)
    optimizer = opt.load_optimizer_from_cfg(model, cfg)
    
    if cfg.train.use_batch_step_lr:
        lr_scheduler = lr_sch.load_batch_lr_scheduler_from_cfg(optimizer, cfg, train_ds)
    else:
        lr_scheduler = lr_sch.load_lr_scheduler_from_cfg(optimizer, cfg)
    

    wandb_log.wandb_init(args, cfg)
    
    dumper.dump_data_holder(data_holder)
    dumper.dump_config(cfg)

    trainer = train.Trainer(cfg, model, (train_ds, val_ds), dumper, 
                            optimizer, lr_scheduler, loss_fn, device, args.small_training)

    # with torch.autograd.detect_anomaly():
    trainer.train()

    wandb_log.wandb_finish()

def main():
    args = get_args(show=True)
    cfg = get_configs(args, show=True)

    if cfg.train.kfold.using:
        kfold(args, cfg)
    else:
        normal(args, cfg)
    
    
if __name__ == '__main__':
    main()
