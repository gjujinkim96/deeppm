import argparse
import models
import train
import wandb_log

from utils import set_seeds, get_device
from DeepPM_utils import *
from experiments.experiment import Experiment
import losses 

import dataset as ds
import data.data_cost as dt
import optimizers as opt
import lr_schedulers as lr_sch

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
    parser.add_argument('--lr_scheduler', required=False, help='For overriding lr_scheduler in train_cfg')
    parser.add_argument('--optimizer', required=False, help='For overriding optimizer in train_cfg')
    parser.add_argument('--lr', required=False, help='For overriding lr in train_cfg')
    parser.add_argument('--batch_size', required=False, help='For overriding batch_size in train_cfg')
    parser.add_argument('--val_batch_size', required=False, help='For overriding val_batch_size in train_cfg')
    parser.add_argument('--clip_grad_norm', required=False, help='For overriding clip_grad_norm in train_cfg')
    parser.add_argument('--n_epochs', required=False, help='For overriding n_epochs in train_cfg')
    parser.add_argument('--checkpoint', required=False, action='store_true', help='For overriding whether to use checkpoint')
    parser.add_argument('--max_len', required=False, help='For overriding max_Len')
    parser.add_argument('--raw_data', required=False, action='store_true', help='For overriding raw_data')
    parser.add_argument('--dim', required=False, help='For overriding dim in model_cfg')
    parser.add_argument('--use_batch_step_lr', required=False, action='store_true', help='For overriding dim in model_cfg')
    parser.add_argument('--hyperparameter_test', required=False, action='store_true', help='For overriding hyperparameter_test')
    parser.add_argument('--hyperparameter_test_mult', required=False, help='For overriding hyperparameter_test_mult')
    parser.add_argument('--short_only', required=False, action='store_true', help='For overriding short_only')
    parser.add_argument('--long_rev', required=False, action='store_true', help='For overriding long_rev')
    parser.add_argument('--fixed_vocab', required=False, action='store_true', help='For overriding fixed_vocab')
    parser.add_argument('--guess_unk', required=False, action='store_true', help='For overriding guess_unk')
    parser.add_argument('--unk_perc', required=False, help='For overriding unk_perc')

    args = parser.parse_args()

    print('################################')
    print(f'Starting {args.experiment_name}')
    print(args)

    train_cfg = train.Config.from_json(args.train_cfg)
    if args.lr is not None:
        train_cfg = train_cfg._replace(lr = float(args.lr))
    if args.lr_scheduler is not None:
        train_cfg = train_cfg._replace(lr_scheduler = args.lr_scheduler)
    if args.batch_size is not None:
        train_cfg = train_cfg._replace(batch_size = int(args.batch_size))
    if args.val_batch_size is not None:
        train_cfg = train_cfg._replace(val_batch_size = int(args.val_batch_size))
    if args.clip_grad_norm is not None:
        train_cfg = train_cfg._replace(clip_grad_norm = float(args.clip_grad_norm))
    if args.n_epochs is not None:
        train_cfg = train_cfg._replace(n_epochs = int(args.n_epochs))
    if args.checkpoint is not None:
        train_cfg = train_cfg._replace(checkpoint = args.checkpoint)
    if args.raw_data is not None:
        train_cfg = train_cfg._replace(raw_data = args.raw_data)
    if args.optimizer is not None:
        train_cfg = train_cfg._replace(optimizer = args.optimizer)
    if args.use_batch_step_lr is not None:
        train_cfg = train_cfg._replace(use_batch_step_lr = args.use_batch_step_lr)
    if args.hyperparameter_test is not None:
        train_cfg = train_cfg._replace(hyperparameter_test = args.hyperparameter_test)
    if args.hyperparameter_test_mult is not None:
        train_cfg = train_cfg._replace(hyperparameter_test_mult = float(args.hyperparameter_test_mult))
    if args.short_only is not None:
        train_cfg = train_cfg._replace(short_only = args.short_only)
    if args.long_rev is not None:
        train_cfg = train_cfg._replace(long_rev = args.long_rev)
    if args.fixed_vocab is not None:
        train_cfg = train_cfg._replace(fixed_vocab = args.fixed_vocab)
        

    print('#############')
    print(train_cfg)
    print()

    set_seeds(train_cfg.seed)

    model_cfg = models.Config.from_json(args.model_cfg)
    if args.model_class is not None:
        model_cfg = model_cfg._replace(model_class = args.model_class)
    if args.max_len is not None:
        model_cfg = model_cfg._replace(max_len = int(args.max_len))
    if args.dim is not None:
        model_cfg = model_cfg._replace(dim = int(args.dim))
    if args.guess_unk is not None:
        model_cfg = model_cfg._replace(guess_unk = args.guess_unk)
    if args.unk_perc is not None:
        model_cfg = model_cfg._replace(unk_perc = float(args.unk_perc))

    expt = Experiment(args.experiment_name, args.experiment_time)
    if expt.check_root_exist() and not args.exp_override:
        print(f'{expt.experiment_root_path()} exist.')
        return 

    data = dt.load_dataset(args.data, small_size=args.small_size,
                           stacked=model_cfg.stacked, only_unique=model_cfg.only_unique,
                           hyperparameter_test=train_cfg.hyperparameter_test,
                            hyperparameter_test_mult=train_cfg.hyperparameter_test_mult,
                            short_only=train_cfg.short_only, rev=train_cfg.long_rev,
                            fixed_vocab=train_cfg.fixed_vocab)
    # data = load_data(args, model_cfg)
    # model_cfg = model_cfg._replace(vocab_size = len(data.token_to_hot_idx))
    model_cfg = model_cfg._replace(pad_idx = data.pad_idx)
    model_cfg = model_cfg._replace(src_idx = data.token_to_hot_idx['<SRCS>'])
    model_cfg = model_cfg._replace(dst_idx = data.token_to_hot_idx['<DSTS>'])
    # model_cfg = model_cfg._replace(dst_idx = data.token_to_hot_idx[data.unk_tok])
    
    print('#############')
    print(model_cfg)
    print()

    

    device = get_device()
    train_ds, test_ds = ds.load_dataset(data, train_cfg, model_cfg)

    model = models.load_model(model_cfg)
    optimizer = opt.load_optimizer(model, train_cfg)

    if train_cfg.use_batch_step_lr:
        total_batch = (len(train_ds) + train_cfg.batch_size - 1) / train_cfg.batch_size
        lr_scheduler = lr_sch.load_batch_step_lr_scheduler(optimizer, train_cfg, total_batch)
    else:
        lr_scheduler = lr_sch.load_lr_scheduler(optimizer, train_cfg)
    loss_fn = losses.load_loss_fn(train_cfg)

    if train_cfg.checkpoint and \
        not (hasattr(model.__class__, 'checkpoint_forward') and callable(getattr(model.__class__, 'checkpoint_forward'))):
        print('Using gradient checkpointing but model support no gradient checkpointing')
        print('Model must implement checkpoint_forward method to use gradient checkpointing')
        return 1

    wandb_log.wandb_init(args, model_cfg, train_cfg, len(train_ds))
    
    dump_model_and_data(model, data, os.path.join(expt.experiment_root_path(), 'predictor.dump'))
    dump_configs(train_cfg, model_cfg, os.path.join(expt.experiment_root_path(), 'config.dump'))

    trainer = train.Trainer(train_cfg, model, (train_ds, test_ds), expt, 
                            optimizer, lr_scheduler, loss_fn, device, args.small_training)

    # with torch.autograd.detect_anomaly():
    trainer.train()

    wandb_log.wandb_finish()

if __name__ == '__main__':
    main()
