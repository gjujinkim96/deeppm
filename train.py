# Copyright 2018 Dong-Hyun Lee, Kakao Brain.

""" Training Config & Helper Classes  """

import os
from typing import NamedTuple
from tqdm.auto import tqdm

import torch
import torch.autograd as autograd
import time
import wandb_log
from torch.utils.data import DataLoader
import multiprocessing
from operator import itemgetter
from utils import correct_regression, seed_worker, get_worker_generator, mape_batch


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

        file_name = os.path.join(self.root_path, 'trained.mdl')
        self.check_point(model,optimizer, lr_scheduler, file_name)

    def save_best(self, model, optimizer, lr_scheduler):
        file_name = os.path.join(self.root_path, 'best.mdl')
        self.check_point(model,optimizer, lr_scheduler, file_name)


    def finish(self, model, optimizer, lr_scheduler):

        self.pbar.close()
        print("Finishing training")

        file_name = os.path.join(self.root_path, 'trained.mdl')
        self.check_point(model,optimizer, lr_scheduler, file_name)

class Trainer(object):
    """ Training Helper Class """
    def __init__(self, cfg, model, ds, expt, optimizer, lr_scheduler, loss_fn, device, small_training):
        self.cfg = cfg # config for training : see class Config
        self.model = model
        self.train_ds, self.val_ds, self.test_ds = ds
        self.expt = expt
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_fn = loss_fn
        self.save_dir = self.expt.experiment_root_path()
        self.device = device # device name

        self.loss_reporter = LossReporter(expt, len(self.train_ds))

        self.clip_grad_norm = cfg.train.clip_grad_norm
        self.small_training = small_training
        # self.cpu_count = multiprocessing.cpu_count()
        self.cpu_count = 0
        self.seed = cfg.train.seed

    def print_final(self, f, x, y):
        if x.shape != ():
            size = x.shape[0]
            for i in range(size):
                f.write('%f,%f ' % (x[i],y[i]))
            f.write('\n')
        else:
            f.write('%f,%f\n' % (x,y))

    class BatchResult:
        def __init__(self):
            self.batch_len = 0

            self.measured = []
            self.prediction = []
            self.inst_lens = []
            self.index = []
            
            self.loss = 0
            self.loss_sum = 0
            self.mape = 0
            self.mape_sum = 0

        def __iadd__(self, other):
            self.batch_len += other.batch_len

            self.measured.extend(other.measured)
            self.prediction.extend(other.prediction)
            self.inst_lens.extend(other.inst_lens)
            self.index.extend(other.index)

            self.loss += other.loss
            self.loss_sum += other.loss_sum

            self.mape += other.mape
            self.mape_sum += other.mape_sum
            return self
        
        def __repr__(self):
            return f'''Batch len: {self.batch_len}
Loss: {self.loss}
'''

    def run_model(self, input, loss_mod=None, is_train=False):
        x = input['x'].to(self.device)
        output = self.model(x)

        y = input['y'].to(self.device)
        loss = self.loss_fn(output, y)

        if loss_mod is not None:
            loss *= loss_mod

        if is_train:
            loss.backward()

        return loss.item(), y.tolist(), output.tolist()
    
    def run_batch(self, batch, is_train=False):
        short, long = itemgetter('short', 'long')(batch)

        result = self.BatchResult()
        short_len = len(short['y'])
        long_len = len(long)
        result.batch_len = short_len + long_len
        result.inst_lens = short['inst_len'] + [item['inst_len'][0] for item in long]
        result.index = short['index'] + [item['index'][0] for item in long]
        
        if short_len > 0:
            loss_mod = short_len / result.batch_len if long_len > 0 else None

            loss, new_y, new_pred = self.run_model(short, loss_mod, is_train)
            result.measured.extend(new_y)
            result.prediction.extend(new_pred)

            mape_score = mape_batch(new_pred, new_y)
            if loss_mod is not None:
                mape_score *= loss_mod
            result.mape += mape_score
            
            result.loss += loss
        
        if long_len > 0:
            for long_item in long:
                loss, new_y, new_pred = self.run_model(long_item, 1/result.batch_len, is_train)
                result.measured.extend(new_y)
                result.prediction.extend(new_pred)
                mape_score = mape_batch(new_pred, new_y)
                mape_score /= result.batch_len

                result.mape += mape_score
                result.loss += loss
                
        
        result.loss_sum = result.loss * result.batch_len
        result.mape_sum = result.mape * result.batch_len
        return result
    
    def validate(self, resultfile, epoch, is_test=False):
        self.model.eval()
        self.model.to(self.device)

        f = open(resultfile,'w')

        ds = self.test_ds if is_test else self.val_ds
        loader = DataLoader(ds, shuffle=False, num_workers=self.cpu_count,
            batch_size=self.cfg.train.val_batch_size, collate_fn=ds.collate_fn)
        epoch_result = self.BatchResult()

        with torch.no_grad():
            for batch in tqdm(loader):
                epoch_result += self.run_batch(batch, is_train=False)
        
    
        epoch_result.loss = epoch_result.loss_sum / epoch_result.batch_len
        epoch_result.mape = epoch_result.mape_sum / epoch_result.batch_len
        correct = correct_regression(epoch_result.prediction, epoch_result.measured, 25)
        f.write(f'loss - {epoch_result.loss}\n')
        f.write(f'{correct}, {epoch_result.batch_len}\n')
        f.close()


        print(f'{"Test" if is_test else "Validate"}: loss - {epoch_result.loss}\n\t{correct}/{epoch_result.batch_len} = {correct / epoch_result.batch_len}\n')
        print()

        if is_test:
            wandb_log.wandb_log_test(epoch_result)
        else:
            wandb_log.wandb_log_val(epoch_result, epoch)

        return epoch_result.mape, correct

    def train(self):
        """ Train Loop """

        best_accr = -1
        generator = get_worker_generator(self.seed)
        resultfile = os.path.join(self.expt.experiment_root_path(), 'validation_results.txt')

        self.model.to(self.device)
        self.optimizer.zero_grad()

        loader = DataLoader(self.train_ds, shuffle=True, num_workers=self.cpu_count, drop_last=True,
                        batch_size=self.cfg.train.batch_size, collate_fn=self.train_ds.collate_fn,
                        worker_init_fn=seed_worker, generator=generator) 
        epoch_len = len(loader)

        for epoch_no in range(self.cfg.train.n_epochs):
            epoch_loss_sum = 0.
            step = 0
            total_correct = 0
            total_cnts = 0
            print(f'using lr: {self.optimizer.param_groups[0]["lr"]}')
            self.loss_reporter.start_epoch(epoch_no + 1) 

            self.model.train()
            
            for idx, batch in enumerate(loader):
                if self.small_training and idx > 100:
                    break
                
                batch_result = self.run_batch(batch, is_train=True)

                if not self.cfg.train.gradient_accumlation.using or (idx + 1) % self.cfg.train.gradient_accumlation.steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                

                wandb_log.wandb_log_train(batch_result, self.lr_scheduler.get_last_lr()[0], 
                            epoch=epoch_no + idx/epoch_len)


                step += 1 
            
                epoch_loss_sum += batch_result.loss_sum
                total_cnts += batch_result.batch_len

                report_batch_len = batch_result.batch_len
                total_correct += correct_regression(batch_result.prediction, batch_result.measured, 25)
                self.loss_reporter.report(report_batch_len, 
                                    batch_result.loss, epoch_loss_sum/step, total_correct/total_cnts)   

                if self.cfg.train.use_batch_step_lr:
                    self.lr_scheduler.step()
            
            epoch_loss_avg = epoch_loss_sum / step
            self.loss_reporter.end_epoch(self.model,self.optimizer, self.lr_scheduler, epoch_loss_avg)


            cur_mape, correct = self.validate(resultfile, epoch_no + 1)

    
            if correct >= best_accr:
                best_accr = correct
                self.loss_reporter.save_best(self.model, self.optimizer, self.lr_scheduler)

            if not self.cfg.train.use_batch_step_lr:
                self.lr_scheduler.step()

        if len(self.test_ds) > 0:
            self.validate(resultfile, epoch_no + 1, is_test=True)
        self.loss_reporter.finish(self.model, self.optimizer, self.lr_scheduler)

