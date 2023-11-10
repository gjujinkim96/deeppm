from handle_inputs import get_test_args
from data import load_data_given_paths
import models
import wandb_log

import torch
from train_loop import validate
from utils import  get_device
from experiment import Experiment, get_default_root_dir

import datasets as ds
import losses as ls

import wandb

def main():
    args = get_test_args(show=True)

    date = args.date
    if date is None:
        root_dir = get_default_root_dir()
        root_model_dir = root_dir.joinpath(args.exp_name)
        date = max([child.name for child in root_model_dir.glob('[0-9][0-9][0-9][0-9]_[0-9][0-9]_[0-9][0-9]') if child.is_dir()])
    exp = Experiment(args.exp_name, date, exp_existing=True)

    if args.type == 'best':
        model_path = exp.best_model_dump
    elif args.type == 'last':
        model_path = exp.trained_model_dump
    elif args.type == 'epoch':
        if args.epoch is None:
            raise ValueError('`--type epoch` option needs --epoch epoch_to_use')
        model_path = exp.epoch_model_dump(args.epoch)
    else:
        raise ValueError('Non-supported args type found.')
    
    if not model_path.exists():
        raise ValueError(f'invalid model_path {model_path}')
    
    cfg = torch.load(exp.config_dump)
    
    if args.idx_dump is not None:
        idx_dict_dump = args.idx_dump
    else:
        idx_dict_dump = exp.idx_dict_dump

    data_holder = load_data_given_paths(cfg, idx_dict_dump, exp.data_mapping_dump, small_size=args.small_size)
    
    device = get_device()

    _, val_ds, _ = ds.load_dataset_from_cfg(data_holder, cfg, show=True)

    model = models.load_model_from_cfg(cfg)

    saved_model = torch.load(model_path, map_location=torch.device('cpu'))
    model_epoch = saved_model['epoch']
    model.load_state_dict(saved_model['model'])

    loss_fn = ls.load_losses_from_cfg(cfg)
    
    run = wandb.init(cfg.log.wandb.project, id='Test', resume=True)
    # wandb_log.wandb_test_init(args, cfg, model_epoch, date)
    
    result = validate(model, val_ds, loss_fn=loss_fn, device=device, batch_size=cfg.train.val_batch_size)


    logging_dict = {
        'epoch': model_epoch+1,
    }

    df = wandb_log.make_df_from_batch_result(result)
    wandb_log.log_cat_mean_error(logging_dict, df, 'val', 'best')

    wandb.log(logging_dict)
    wandb_log.wandb_finish()


if __name__ == '__main__':
    main()
