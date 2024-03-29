from handle_inputs import get_test_only_args
from data import load_test_only_data_given_paths
import models
import wandb_log

import torch
from train_loop import validate
from utils import  get_device, correct_regression
from experiment import Experiment, get_default_root_dir

import datasets as ds
import losses as ls


def main():
    args = get_test_only_args(show=True)

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
    cfg.log.wandb.project = args.wandb_project

    data_holder = load_test_only_data_given_paths(cfg, args.data_path, exp.data_mapping_dump, args.small_size)
    
    device = get_device()

    _, _, test_ds = ds.load_dataset_from_cfg(data_holder, cfg, show=True)

    model = models.load_model_from_cfg(cfg)

    saved_model = torch.load(model_path, map_location=torch.device('cpu'))
    model_epoch = saved_model['epoch']
    model.load_state_dict(saved_model['model'])
    print('model loaded')

    loss_fn = ls.load_losses_from_cfg(cfg)
    
    wandb_log.wandb_test_init(args, cfg, model_epoch, date)
    
    result = validate(model, test_ds, loss_fn=loss_fn, device=device, batch_size=cfg.train.val_batch_size)

    test_correct = correct_regression(result.prediction, result.measured, 25)
    print(f'Test: loss - {result.loss}\n\t{test_correct}/{result.batch_len} = {test_correct / result.batch_len}\n')
    print()

    wandb_log.wandb_log_test(result)
    wandb_log.wandb_finish()


if __name__ == '__main__':
    main()
