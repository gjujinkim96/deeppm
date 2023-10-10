from handle_inputs import get_test_args
from data import load_data_given_paths
import models
import wandb_log

import torch
from train_loop import validate
from utils import  get_device, correct_regression
from experiment import Experiment

import datasets as ds
import losses as ls


def main():
    args = get_test_args(show=True)

    exp = Experiment(args.exp_name, args.date, exp_existing=True)

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
    data = load_data_given_paths(cfg, idx_dict_dump, exp.data_mapping_dump, small_size=args.small_size)
    
    device = get_device()

    _, _, test_ds = ds.load_dataset_from_cfg(data, cfg, show=True)

    model = models.load_model_from_cfg(cfg)

    saved_model = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(saved_model['model'])

    loss_fn = ls.load_losses_from_cfg(cfg)
    

    wandb_log.wandb_test_init(args, cfg)
    
    result = validate(model, test_ds, loss_fn=loss_fn, device=device, batch_size=cfg.train.val_batch_size)

    val_correct = correct_regression(result.prediction, result.measured, 25)
    print(f'Validate: loss - {result.loss}\n\t{val_correct}/{result.batch_len} = {val_correct / result.batch_len}\n')
    print()

    wandb_log.wandb_log_test(result)
    wandb_log.wandb_finish()


if __name__ == '__main__':
    main()