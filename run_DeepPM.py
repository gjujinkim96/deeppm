import argparse
import torch
import torch.optim as optim
import torch.nn as nn
import models
import train
# import optim

from utils import set_seeds, get_device
from DeepPM_utils import *
from experiments.experiment import Experiment
from losses import mse_loss, LogCoshLoss

from dataset import BasicBlockDataset

def main():
    # type: () -> None
    parser = argparse.ArgumentParser()

    # data arguments
    parser.add_argument('--data', required=True, help='The data file to load from')
    parser.add_argument('--train_cfg', required=True, help='Configuration for train')
    parser.add_argument('--model_cfg', required=True, help='Configuration of model')
    parser.add_argument('--experiment-name', required=True, help='Name of the experiment to run')
    parser.add_argument('--experiment-time', required=True, help='Time the experiment was started at')
    parser.add_argument('--small_size', required=False, default=False, help='For quick test')

    args = parser.parse_args()

    train_cfg = train.Config.from_json(args.train_cfg)
    set_seeds(train_cfg.seed)

    data = load_data(args.data, small_size=args.small_size)

    model_cfg = models.Config.from_json(args.model_cfg)
    model_cfg.set_vocab_size(len(data.token_to_hot_idx))
    model_cfg.set_pad_idx(data.pad_idx)

    train_ds = BasicBlockDataset(data.train, model_cfg.max_len)
    test_ds = BasicBlockDataset(data.test, model_cfg.max_len)

    expt = Experiment(args.experiment_name, args.experiment_time)

    device = get_device()
    model = models.DeepPM(model_cfg)
    optimizer = optim.Adam(model.parameters(), lr=train_cfg.lr)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, min_lr=train_cfg.lr * 0.001)

    loss_fn = LogCoshLoss()
    
    dump_model_and_data(model, data, os.path.join(expt.experiment_root_path(), 'predictor.dump'))

    trainer = train.Trainer(train_cfg, model, (train_ds, test_ds), expt, 
                            optimizer, lr_scheduler, loss_fn, device)
    trainer.train()

if __name__ == '__main__':
    main()
