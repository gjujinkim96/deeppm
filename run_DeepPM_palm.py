import argparse
import torch
import torch.optim as optim
import torch.nn as nn
import models
import train_palm as train
import wandb

from utils import set_seeds, get_device
from DeepPM_utils import *
from experiments.experiment import Experiment
from losses import mse_loss, LogCoshLoss

from palmdataset import PalmDataset
import palmtree.eval_utils as eval_utils
import os
from config import *
from torch import nn
# from scipy.ndimage.filters import gaussian_filter1d
from torch.autograd import Variable
import torch
import numpy as np

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

    args = parser.parse_args()

    train_cfg = train.Config.from_json(args.train_cfg)
    set_seeds(train_cfg.seed)

    data = load_data(args.data, small_size=args.small_size)

    model_cfg = models.Config.from_json(args.model_cfg)
    model_cfg.set_vocab_size(len(data.token_to_hot_idx))
    model_cfg.set_pad_idx(data.pad_idx)

    wandb.init(
        project='deeppm',
        config={
            "dim": model_cfg.dim,
	        "dim_ff": model_cfg.dim_ff,
	        "n_layers": model_cfg.n_layers,
	        "n_heads": model_cfg.n_heads,
	        "max_len": model_cfg.max_len,

            "batch_size": train_cfg.batch_size,
            "lr": train_cfg.lr,
            "n_epochs": train_cfg.n_epochs,
        }
    )

    device = get_device()

    train_ds = torch.load('encoded/encoded_train.pkl')
    test_ds = torch.load('encoded/encoded_test.pkl')
    # palmtree = eval_utils.UsableTransformer(
    #     model_path="./palmtree/palmtree/transformer.ep19", vocab_path="./palmtree/palmtree/vocab")
    # train_ds = PalmDataset(data.train, palmtree)
    # test_ds = PalmDataset(data.test, palmtree)
    # del palmtree

    expt = Experiment(args.experiment_name, args.experiment_time)

    
    model = models.PalmDeepPM(model_cfg)
    optimizer = optim.Adam(model.parameters(), lr=train_cfg.lr)
    lr_scheduler = optim.lr_scheduler.LinearLR(optimizer)

    loss_fn = mse_loss
    
    dump_model_and_data(model, data, os.path.join(expt.experiment_root_path(), 'predictor.dump'))

    trainer = train.Trainer(train_cfg, model, (train_ds, test_ds), expt, 
                            optimizer, lr_scheduler, loss_fn, device, args.small_training)
    trainer.train()

    wandb.finish()

if __name__ == '__main__':
    main()
