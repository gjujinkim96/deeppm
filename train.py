# Copyright 2018 Dong-Hyun Lee, Kakao Brain.

""" Training Config & Helper Classes  """

import torch
import torch.autograd as autograd

import wandb_log
from torch.utils.data import DataLoader

import multiprocessing
from utils import correct_regression, seed_worker, get_worker_generator


from train_loop import run_batch, validate
from loss_reporter import LossReporter

class Trainer(object):
    """ Training Helper Class """
    def __init__(self, cfg, model, ds, dumper, optimizer, lr_scheduler, loss_fn, device, small_training):
        self.cfg = cfg 
        self.model = model
        self.train_ds, self.val_ds, self.test_ds = ds
        self.dumper = dumper
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_fn = loss_fn
        self.device = device 

        if self.cfg.train.cpu_count is None:
            self.cpu_count = multiprocessing.cpu_count()
        else:
            self.cpu_count = self.cfg.train.cpu_count

        self.loss_reporter = LossReporter(len(self.train_ds))
        self.small_training = small_training

    def train(self):
        # Setup
        best_correct = -1
        generator = get_worker_generator(self.cfg.train.seed)

        self.model.to(self.device)
        self.optimizer.zero_grad()

        loader = DataLoader(self.train_ds, shuffle=True, num_workers=self.cpu_count,
                        batch_size=self.cfg.train.batch_size, collate_fn=self.train_ds.collate_fn,
                        worker_init_fn=seed_worker, generator=generator) 
        epoch_len = len(loader)
        save_epoch = self.cfg.train.save_epoch

        # train loop
        for epoch_no in range(self.cfg.train.n_epochs):
            print(f'using lr: {self.optimizer.param_groups[0]["lr"]}')
            self.loss_reporter.start_epoch(epoch_no + 1) 

            self.model.train()
            
            for idx, batch in enumerate(loader):
                if self.small_training and idx > 10:
                    break

                batch_result = run_batch(batch, self.model, 
                        is_train=True, loss_fn=self.loss_fn, device=self.device)

                if not self.cfg.train.gradient_accumlation.using or (idx + 1) % self.cfg.train.gradient_accumlation.steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.clip_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                wandb_log.wandb_log_train(batch_result, self.lr_scheduler.get_last_lr()[0], 
                            epoch=epoch_no + idx/epoch_len)

                self.loss_reporter.report(batch_result)

                if self.cfg.train.use_batch_step_lr:
                    self.lr_scheduler.step()
            
            self.loss_reporter.end_epoch()
            self.loss_reporter.log(self.dumper)
            if save_epoch != -1:
                self.dumper.save_trained_model(epoch_no+1, self.model, self.optimizer, self.lr_scheduler)
            if save_epoch > 0:
                if (epoch_no + 1) % save_epoch == 0:
                    self.dumper.save_epoch_model(epoch_no+1, self.model, self.optimizer, self.lr_scheduler)

            # Validation
            val_result = validate(self.model, self.val_ds, 
                        loss_fn=self.loss_fn, device=self.device, batch_size=self.cfg.train.val_batch_size)

            val_correct = correct_regression(val_result.prediction, val_result.measured, 25)
            print(f'Validate: loss - {val_result.loss}\n\t{val_correct}/{val_result.batch_len} = {val_correct / val_result.batch_len}\n')
            print()
            wandb_log.wandb_log_val(val_result, epoch_no + 1)
            self.dumper.append_to_val_result(val_result, val_correct)
    
            if save_epoch != -1 and val_correct >= best_correct:
                best_correct = val_correct
                self.dumper.save_best_model(epoch_no+1, self.model, self.optimizer, self.lr_scheduler)

            if not self.cfg.train.use_batch_step_lr:
                self.lr_scheduler.step()

        self.loss_reporter.finish()

