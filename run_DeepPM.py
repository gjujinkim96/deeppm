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

    args = parser.parse_args()

    train_cfg = train.Config.from_json(args.train_cfg)
    print('#############')
    print(train_cfg)
    print()

    set_seeds(train_cfg.seed)

    model_cfg = models.Config.from_json(args.model_cfg)

    data = load_data(args, model_cfg)
    model_cfg.set_vocab_size(len(data.token_to_hot_idx))
    model_cfg.set_pad_idx(data.pad_idx)

    print('#############')
    print(model_cfg)
    print()

    

    device = get_device()
    
    train_ds, test_ds = ds.load_dataset(data, train_cfg, model_cfg)

    expt = Experiment(args.experiment_name, args.experiment_time)


    model = models.load_model(model_cfg)
    optimizer = opt.load_optimizer(model, train_cfg)
    lr_scheduler = lr_sch.load_lr_scheduler(optimizer, train_cfg)
    loss_fn = losses.load_loss_fn(train_cfg)

    wandb_log.wandb_init(args, model_cfg, train_cfg, len(train_ds))
    
    dump_model_and_data(model, data, os.path.join(expt.experiment_root_path(), 'predictor.dump'))

    trainer = train.Trainer(train_cfg, model, (train_ds, test_ds), expt, 
                            optimizer, lr_scheduler, loss_fn, device, args.small_training)

    # with torch.autograd.detect_anomaly():
    trainer.train()

    wandb_log.wandb_finish()

if __name__ == '__main__':
    main()
