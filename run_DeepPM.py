from handle_inputs import get_args, get_configs
from data import load_data_from_cfg
import models
import train
import wandb_log

from utils import set_seeds, get_device
from experiment import Experiment

import datasets as ds
import data.data_cost as dt
import optimizers as opt
import lr_schedulers as lr_sch
import losses as ls

from dumper import Dumper


def main():
    args = get_args(show=True)
    cfg = get_configs(args, show=True)
    
    set_seeds(cfg.train.seed)
    
    expt = Experiment(args.exp_name, exp_override=args.exp_override)
    expt.restart()
    dumper = Dumper(expt)

    data = load_data_from_cfg(args.small_size, cfg)

    device = get_device()

    train_ds, val_ds, test_ds = ds.load_dataset_from_cfg(data, cfg, show=True)
    model = models.load_model_from_cfg(cfg)
    loss_fn = ls.load_losses_from_cfg(cfg)
    optimizer = opt.load_optimizer_from_cfg(model, cfg)
    
    if cfg.train.use_batch_step_lr:
        lr_scheduler = lr_sch.load_batch_lr_scheduler_from_cfg(optimizer, cfg, train_ds)
    else:
        lr_scheduler = lr_sch.load_lr_scheduler_from_cfg(optimizer, cfg)
    

    wandb_log.wandb_init(args, cfg)
    
    dumper.dump_data_mapping(data.dump_dataset_params())
    dumper.dump_config(cfg)
    dumper.dump_idx_dict(data)

    trainer = train.Trainer(cfg, model, (train_ds, val_ds, test_ds), dumper, 
                            optimizer, lr_scheduler, loss_fn, device, args.small_training)

    # with torch.autograd.detect_anomaly():
    trainer.train()

    wandb_log.wandb_finish()

if __name__ == '__main__':
    main()
