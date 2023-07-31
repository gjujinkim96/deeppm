import argparse
import torch

import torch.nn as nn
import models
import train
import wandb_log

from utils import set_seeds, get_device
from DeepPM_utils import *
from experiments.experiment import Experiment
import losses 

import dataset as ds
import optimizers as opt
import lr_schedulers as lr_sch
from dataset import BasicBlockDataset, StackedBlockDataset

def main():
    # type: () -> None
    parser = argparse.ArgumentParser()

    # data arguments
    parser.add_argument('--data', required=True, help='The data file to load from')
    parser.add_argument('--train_cfg', required=True, help='Configuration for train')
    parser.add_argument('--model_cfg', required=True, help='Configuration of model')
    parser.add_argument('--experiment-name', required=True, help='Name of the experiment to run')
    parser.add_argument('--experiment-time', required=True, help='Time the experiment was started at')
    parser.add_argument('--small_size', required=False, action='store_true', help='For quick test')
    parser.add_argument('--small_training', required=False, action='store_true', help='For quick validation test')
    parser.add_argument('--wandb_disabled', required=False, action='store_true', help='For turning of wandb logging')
    parser.add_argument('--exp_override', required=False, action='store_true', help='For overriding run')
    parser.add_argument('--model_class', required=False, help='For overriding model_class in model_cfg')
    parser.add_argument('--lr', required=False, help='For overriding lr in train_cfg')
    parser.add_argument('--clip_grad_norm', required=False, help='For overriding clip_grad_norm in train_cfg')
    parser.add_argument('--n_epochs', required=False, help='For overriding n_epochs in train_cfg')
    parser.add_argument('--checkpoint', required=False, action='store_true', help='For overriding whether to use checkpoint')
    parser.add_argument('--max_len', required=False, help='For overriding max_Len')
    parser.add_argument('--raw_data', required=False, action='store_true', help='For overriding raw_data')

    args = parser.parse_args()

    print(args)

    train_cfg = train.Config.from_json(args.train_cfg)
    if args.lr is not None:
        train_cfg = train_cfg._replace(lr = float(args.lr))
    if args.clip_grad_norm is not None:
        train_cfg = train_cfg._replace(clip_grad_norm = float(args.clip_grad_norm))
    if args.n_epochs is not None:
        train_cfg = train_cfg._replace(n_epochs = int(args.n_epochs))
    if args.checkpoint is not None:
        train_cfg = train_cfg._replace(checkpoint = args.checkpoint)
    if args.raw_data is not None:
        train_cfg = train_cfg._replace(raw_data = args.raw_data)
        

    print('#############')
    print(train_cfg)
    print()

    set_seeds(train_cfg.seed)

    model_cfg = models.Config.from_json(args.model_cfg)
    if args.model_class is not None:
        model_cfg = model_cfg._replace(model_class = args.model_class)
    if args.max_len is not None:
        model_cfg = model_cfg._replace(max_len = int(args.max_len))

    expt = Experiment(args.experiment_name, args.experiment_time)
    if expt.check_root_exist() and not args.exp_override:
        print(f'{expt.experiment_root_path()} exist.')
        return 

    data = load_data(args, model_cfg)
    model_cfg = model_cfg._replace(vocab_size = len(data.token_to_hot_idx))
    model_cfg = model_cfg._replace(pad_idx = data.pad_idx)
    
    print('#############')
    print(model_cfg)
    print()

    

    device = get_device()
    train_ds, test_ds = ds.load_dataset(data, train_cfg, model_cfg)

    model = models.load_model(model_cfg)
    optimizer = opt.load_optimizer(model, train_cfg)
    lr_scheduler = lr_sch.load_lr_scheduler(optimizer, train_cfg)
    loss_fn = losses.load_loss_fn(train_cfg)

    if train_cfg.checkpoint and \
        not (hasattr(model.__class__, 'checkpoint_forward') and callable(getattr(model.__class__, 'checkpoint_forward'))):
        print('Using gradient checkpointing but model support no gradient checkpointing')
        print('Model must implement checkpoint_forward method to use gradient checkpointing')
        return 1

    wandb_log.wandb_init(args, model_cfg, train_cfg, len(train_ds))
    
    dump_model_and_data(model, data, os.path.join(expt.experiment_root_path(), 'predictor.dump'))

    trainer = train.Trainer(train_cfg, model, (train_ds, test_ds), expt, 
                            optimizer, lr_scheduler, loss_fn, device, args.small_training)

    # with torch.autograd.detect_anomaly():
    trainer.train()

    wandb_log.wandb_finish()

if __name__ == '__main__':
    main()
