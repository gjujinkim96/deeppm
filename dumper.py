import torch

def make_state_dict(epoch_no, model, optimizer, lr_scheduler):
    return {
        'epoch': epoch_no,
        'model': model.state_dict(),
        'optimizer':optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
    }

class Dumper:
    def __init__(self, experiment):
        self.experiment = experiment

    def dump_config(self, config):
        torch.save(config, self.experiment.config_dump)

    def dump_data_holder(self, data_holder):
        torch.save(data_holder.converter.dump_params(), self.experiment.data_mapping_dump)

        idx_dict = {
            'train': [datum.code_id for datum in data_holder.train],
            'val': [datum.code_id for datum in data_holder.val],
            'test': [datum.code_id for datum in data_holder.test],
        }
        torch.save(idx_dict, self.experiment.idx_dict_dump)

    def append_to_loss_log(self, epoch_no, time, loss, accr):
        with open(self.experiment.loss_report_log, 'a') as f:
            f.write(f'{epoch_no}\t{time:.6f}\t{loss:.6f}\t{accr:.6f}\n')

    def append_to_val_result(self, val_result, val_correct):
        with open(self.experiment.validation_results, 'a') as f:
            f.write(f'{val_result.loss:.6f},{val_correct},{val_result.batch_len}\n')
    
    def save_best_model(self, epoch_no, model, optimizer, lr_scheduler):
        torch.save(make_state_dict(epoch_no, model, optimizer, lr_scheduler),
                self.experiment.best_model_dump)
        
    def save_trained_model(self, epoch_no, model, optimizer, lr_scheduler):
        torch.save(make_state_dict(epoch_no, model, optimizer, lr_scheduler),
                self.experiment.trained_model_dump)

    def save_epoch_model(self, epoch_no, model, optimizer, lr_scheduler):
        torch.save(make_state_dict(epoch_no, model, optimizer, lr_scheduler),
                self.experiment.epoch_model_dump(epoch_no))
