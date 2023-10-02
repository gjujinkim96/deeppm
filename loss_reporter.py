import time
from tqdm.auto import tqdm
from utils import correct_regression

class LossReporter(object):
    def __init__(self, n_data_points):
        self.n_datapoints = n_data_points
        self.start_time = time.time()
        self.epoch_no = 0

        self.loss = 1.0        
        self.step = 0
        self.epoch_loss_sum = 0
        self.total_cnts = 0
        self.total_correct = 0

    @property
    def avg_loss(self):
        if self.step == 0:
            return 10
        return self.epoch_loss_sum/self.step

    @property
    def avg_accuracy(self):
        if self.total_cnts == 0:
            return 0
        return self.total_correct/self.total_cnts
    
    def format_loss(self):
        return f'Epoch {self.epoch_no}, Loss: {self.loss:.2f}, {self.avg_loss:.2f}, Accuracy: {self.avg_accuracy:.2f}'

    def start_epoch(self, epoch_no):
        self.epoch_no = epoch_no

        self.step = 0
        self.epoch_loss_sum = 0
        self.total_cnts = 0
        self.total_correct = 0

        self.pbar = tqdm(desc=self.format_loss(), total=self.n_datapoints)

    def report(self, batch_result):
        self.step += 1
        self.loss = batch_result.loss
        self.epoch_loss_sum += batch_result.loss_sum
        self.total_cnts += batch_result.batch_len
        self.total_correct += correct_regression(batch_result.prediction, batch_result.measured, 25)

        desc = self.format_loss()
        self.pbar.set_description(desc)
        self.pbar.update(batch_result.batch_len)


    def end_epoch(self):
        self.pbar.close()

    def log(self, dumper):
        dumper.append_to_loss_log(self.epoch_no, time.time()-self.start_time, self.avg_loss, self.avg_accuracy)

    def finish(self):
        self.pbar.close()
        print("Finishing training")
