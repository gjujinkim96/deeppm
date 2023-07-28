# Copyright 2018 Dong-Hyun Lee, Kakao Brain.

""" Training Config & Helper Classes  """

import os
import json
from typing import NamedTuple
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import time
import losses as ls
import wandb
from torch.utils.data import DataLoader
import multiprocessing
import pandas as pd
# import plotting


def sort_by_order(values, order):
    tmp = list(zip(values, order))
    tmp.sort(key=lambda x: x[1])
    return [x[0] for x in tmp]

class Config(NamedTuple):
    """ Hyperparameters for training """
    seed: int = 3431 # random seed
    batch_size: int = 32
    lr: int = 5e-5 # learning rate
    n_epochs: int = 10 # the number of epoch
    # `warm up` period = warmup(0.1)*total_steps
    # linearly increasing learning rate from zero to the specified value(5e-5)
    warmup: float = 0.001
    #save_steps: int = 100 # interval for saving model
    #total_steps: int = 100000 # total number of steps to train

    @classmethod
    def from_json(cls, file): # load config from json file
        return cls(**json.load(open(file, "r")))

class LossReporter(object):
    def __init__(self, experiment, n_data_points):
        # type: (Experiment, int, tr.Train) -> None

        self.experiment = experiment
        self.n_datapoints = n_data_points
        self.start_time = time.time()

        self.loss = 1.0
        self.avg_loss = 1.0
        self.epoch_no = 0
        self.total_processed_items = 0
        self.epoch_processed_items = 0
        self.accuracy = 0.0

        self.last_report_time = 0.0
        self.last_save_time = 0.0

        self.root_path = self.experiment.experiment_root_path()

        try:
            os.makedirs(self.root_path)
        except OSError:
            pass

        self.loss_report_file = open(os.path.join(self.root_path, 'loss_report.log'), 'w',1)
        self.pbar = tqdm(desc = self.format_loss(), total=self.n_datapoints)

    def format_loss(self):

        return 'Epoch {}, Loss: {:.2}, {:.2}, Accuracy: {:.2}'.format(
                self.epoch_no,
                self.loss,
                self.avg_loss,
                self.accuracy
        )

    def start_epoch(self, epoch_no):
        
        self.epoch_no = epoch_no
        self.epoch_processed_items = 0
        self.accuracy = 0.0

        self.pbar.close()
        self.pbar = tqdm(desc=self.format_loss(), total=self.n_datapoints)

    def report(self, n_items, loss, avg_loss, t_accuracy):

        self.loss = loss
        self.avg_loss = avg_loss
        #self.accuracy = (self.accuracy * self.epoch_processed_items + t_accuracy * n_items) / (self.epoch_processed_items + n_items)
        self.accuracy = t_accuracy
        self.epoch_processed_items += n_items
        self.total_processed_items += n_items

        desc = self.format_loss()
        self.pbar.set_description(desc)
        self.pbar.update(n_items)

    def check_point(self, model, optimizer, lr_scheduler, file_name):

        state_dict = {
            'epoch': self.epoch_no,
            'model': model.state_dict(),
            'optimizer':optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
        }
            
        try: 
            os.makedirs(os.path.dirname(file_name))
        except OSError:
            pass

        torch.save(state_dict, file_name) 

    def end_epoch(self, model, optimizer, lr_scheduler, loss):
        
        self.loss = loss

        t = time.time()
        message = '\t'.join(map(str, (
            self.epoch_no,
            t - self.start_time,
            self.loss,
            self.accuracy,
        )))
        self.loss_report_file.write(message + '\n')

        file_name = os.path.join(self.experiment.checkpoint_file_dir(),'{}.mdl'.format(self.epoch_no))
        self.check_point(model,optimizer, lr_scheduler, file_name)


    def finish(self, model, optimizer, lr_scheduler):

        self.pbar.close()
        print("Finishing training")

        file_name = os.path.join(self.root_path, 'trained.mdl')
        self.check_point(model,optimizer, lr_scheduler, file_name)


class Trainer(object):
    """ Training Helper Class """
    def __init__(self, train_cfg, model, ds, expt, optimizer, lr_scheduler, loss_fn, device, small_training):
        self.train_cfg = train_cfg # config for training : see class Config
        self.model = model
        self.train_ds, self.test_ds = ds
        self.expt = expt
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.save_dir = self.expt.experiment_root_path()
        self.device = device # device name

        self.loss_fn = loss_fn
        self.loss_reporter = LossReporter(expt, len(self.train_ds))

        self.tolerance = 25.
        self.small_training = small_training
        self.cpu_count = multiprocessing.cpu_count()


    def correct_regression(self, x, y):
        if x.shape != ():
            x = x[-1]
            y = y[-1]

        percentage = torch.abs(x - y) * 100.0 / (y + 1e-3)

        if percentage < self.tolerance:
            self.correct += 1

    def torch_correct_regression(self, x, y):
        percentage = torch.abs(x - y) * 100.0 / (y + 1e-3)
        return sum(percentage < self.tolerance)


    def print_final(self, f, x, y):
        if x.shape != ():
            size = x.shape[0]
            for i in range(size):
                f.write('%f,%f ' % (x[i],y[i]))
            f.write('\n')
        else:
            f.write('%f,%f\n' % (x,y))

    def validate(self, resultfile):
        self.model.eval()
        self.model.to(self.device)

        f = open(resultfile,'w')

        correct = 0
        with torch.no_grad():
            loader = DataLoader(self.test_ds, shuffle=False, num_workers=self.cpu_count,
                        batch_size=self.train_cfg.batch_size, collate_fn=self.test_ds.block_collate_fn)
            
            total_correct = 0
            total_total = 0
            total_loss_sum = 0
            
            # offset = 0
            answer = []
            predict = []
            # order = []


            for x, y, mask in tqdm(loader):
                total = len(y)
                total_total += total
                x = x.to(self.device)
                y = y.to(self.device)
                mask = mask.to(self.device)

                output = self.model(x, mask)
                loss = self.loss_fn(output, y)

                cur_correct = self.torch_correct_regression(output, y)
                total_correct += cur_correct

                answer.extend(y.tolist())
                predict.extend(output.tolist())

                batch_loss = loss.item()
                total_loss_sum += batch_loss * total

        ret_loss = total_loss_sum/total_total
        f.write(f'loss - {ret_loss}\n')
        f.write(f'{correct}, {len(self.test_ds)}\n')
        print(f'Validate: loss - {ret_loss}\n\t{total_correct}/{len(self.test_ds)} = {total_correct / total_total}\n')
        print()

        data = [[x, y] for (x, y) in zip(answer, predict)]
        table = wandb.Table(data=data, columns = ["answer", "predict"])
        wandb.log({
                    "val_loss": ret_loss,
                    "val_batch_correct": total_correct,
                    "val_batch_accuracy": total_correct / total_total,
                    "comp": wandb.plot.scatter(table, "answer", "predict")
                })

        f.close()
        return ret_loss

    def train(self):
        """ Train Loop """
        resultfile = os.path.join(self.expt.experiment_root_path(), 'validation_results.txt')

        self.model.to(self.device)
        wandb.watch(self.model, log_freq=500)

        loader = DataLoader(self.train_ds, shuffle=True, num_workers=self.cpu_count, 
                        batch_size=self.train_cfg.batch_size, collate_fn=self.train_ds.block_collate_fn)

        # with autograd.detect_anomaly(True):
        for epoch_no in range(self.train_cfg.n_epochs):
            epoch_loss_sum = 0.
            step = 0
            # print(f'using lr: {self.optimizer.param_groups[0]["lr"]}')
            self.loss_reporter.start_epoch(epoch_no + 1) 

            self.model.train()
            
            total_cnts = 0
            total_correct = 0
            epoch_loss_sum = 0
            for idx, (x, y, mask) in enumerate(loader):
                if self.small_training and idx > 100:
                    break

                total = len(y)

                x = x.to(self.device)
                y = y.to(self.device)
                mask = mask.to(self.device)

                output = self.model(x, mask)
                loss = self.loss_fn(output, y) 
                loss.backward()

                cur_correct = self.torch_correct_regression(output, y)
                total_correct += cur_correct


                torch.nn.utils.clip_grad_norm_(self.model.parameters(), .2)
                
                for param in self.model.parameters():
                    if param.grad is None:
                        continue

                    if torch.isnan(param.grad).any():
                        print("BAD: isnan found in grad")
                        self.loss_reporter.finish(self.model, self.optimizer, self.lr_scheduler)
                        return
                
                self.optimizer.step()


                step += 1
                batch_loss = loss.item()
                epoch_loss_sum += batch_loss
                total_cnts += total
                self.loss_reporter.report(total, batch_loss, epoch_loss_sum/step, total_correct/total_cnts)   

                
                wandb.log({
                    "loss": batch_loss,
                    "batch_size": total,
                    'batch_correct': cur_correct,
                    "batch_accuracy": cur_correct / total,
                    "lr": self.lr_scheduler.get_last_lr()[0],
                })

            epoch_loss_avg = epoch_loss_sum / step
            self.loss_reporter.end_epoch(self.model,self.optimizer, self.lr_scheduler, epoch_loss_avg)

            val_loss = self.validate(resultfile)
            self.lr_scheduler.step()
        self.loss_reporter.finish(self.model,self.optimizer, self.lr_scheduler)
