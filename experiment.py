from pathlib import Path
import datetime
import shutil

HOME = Path(__file__).parent

class Experiment:
    def __init__(self, name, time=None, exp_override=True):
        if time is None:
            now = datetime.datetime.now()
            time = f'{now.month:02}{now.day:02}'

        self.root_dir = Path(HOME, 'saved', name, time)
        
        if not exp_override and self.root_dir.is_dir():
            raise ValueError(f'{self.root_dir} exist when it should not.')
        
        self.root_dir.mkdir(parents=True, exist_ok=True)

        self.config_dump = self.root_dir.joinpath('config.dump')
        self.data_mapping_dump = self.root_dir.joinpath('data_mapping.dump')
        self.idx_dict_dump = self.root_dir.joinpath('idx_dict.dump')
        self.loss_report_log = self.root_dir.joinpath('loss_report.csv')
        self.validation_results = self.root_dir.joinpath('validation_results.csv')

        self.trained_model_dump = self.root_dir.joinpath('trained.mdl')
        self.best_model_dump = self.root_dir.joinpath('best.mdl')

        self.model_dir = self.root_dir.joinpath('save_by_epoch')
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def epoch_model_dump(self, epoch_no):
        return self.model_dir.joinpath(f'{epoch_no}.mdl')
    
    def restart(self):
        shutil.rmtree(self.root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.loss_report_log.touch()
        with open(self.loss_report_log, 'a') as f:
            f.write('epoch\ttime\t\tloss\t\taccuracy\n')

        self.validation_results.touch()
        with open(self.validation_results, 'a') as f:
            f.write('loss,correct,batch\n')
