import torch
from tqdm.auto import tqdm
import argparse
from pathlib import Path
from data import load_data_from_cfg
from models import load_model_from_cfg
from torch.utils.data import DataLoader
import dataset as ds
import multiprocessing
from train import Trainer
from operator import itemgetter
from utils import correct_regression, mape_batch

parser = argparse.ArgumentParser()

parser.add_argument('model_name')
parser.add_argument('run_date', nargs='?', default=None)

args = parser.parse_args()

model_class_root = Path('saved', args.model_name)

if args.run_date is None:
    max_value = -1
    model_root = None
    for x in model_class_root.glob('[0-9]*'):
        date = int(x.parts[-1])
        if max_value < date:
            max_value = date
            model_root = x
    if model_root is None:
        raise ValueError('No date folder is found for given model')
else:
    model_root = model_class_root.joinpath(args.run_date)
    


# model_root = 'model_result/two_baseline_mape/0823/'
config_file = model_root.joinpath('config.dump')
if not config_file.is_file():
    raise ValueError('No config file is found')

model_file = model_root.joinpath('best.mdl')
if not model_file.is_file():
    model_file = model_root.joinpath('trained.mdl')
if not model_file.is_file():
    raise ValueError('No model file is found')

dm_file = model_root.joinpath('data_mapping.dump')
if not model_file.is_file():
    print('No data_mapping.dump is found. Using None instead.')
    dm = None
else:
    dm = torch.load(dm_file)[0]

idx_dict_file = model_root.joinpath('idx_dict.dump')
if not idx_dict_file.is_file():
    print('No idx_dict.dump is found. Using None instead.')
    idx_dict = None
else:
    idx_dict = torch.load(idx_dict_file)

cfg = torch.load(config_file)
data = load_data_from_cfg(False, cfg, dm, idx_dict)

model_dump = torch.load(model_file, map_location='cpu')

model = load_model_from_cfg(cfg)
model.load_state_dict(model_dump['model'])


train_ds, val_ds, test_ds = ds.load_dataset_from_cfg(data, cfg, show=True)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cpu_count = multiprocessing.cpu_count()

model.eval()
model.to(device)

loader = DataLoader(val_ds, shuffle=False, num_workers=cpu_count,
    batch_size=cfg.train.val_batch_size, collate_fn=val_ds.collate_fn)

epoch_result = Trainer.BatchResult()

with torch.no_grad():
    for batch in tqdm(loader):
        short, long = itemgetter('short', 'long')(batch)

        result = Trainer.BatchResult()
        short_len = len(short['y'])
        long_len = len(long)
        result.batch_len = short_len + long_len
        result.inst_lens = short['inst_len'] + [item['inst_len'][0] for item in long]
        result.index = short['index'] + [item['index'][0] for item in long]
        
        if short_len > 0:
            loss_mod = short_len / result.batch_len if long_len > 0 else None

            short['x'] = short['x'].to(device)
            short['y'] = short['y'].to(device)

            loss, new_y, new_pred = model.run(short, loss_mod, False)
            result.measured.extend(new_y)
            result.prediction.extend(new_pred)

            mape_score = mape_batch(new_pred, new_y)
            if loss_mod is not None:
                mape_score *= loss_mod
            result.mape += mape_score
            for k in loss:
                result.loss[k] += loss[k]
        
        if long_len > 0:
            for long_item in long:
                long_item['x'] = long_item['x'].to(device)
                long_item['y'] = long_item['y'].to(device)

                loss, new_y, new_pred = model.run(long_item, 1/result.batch_len, False)
                result.measured.extend(new_y)
                result.prediction.extend(new_pred)
                mape_score = mape_batch(new_pred, new_y)
                mape_score /= result.batch_len

                result.mape += mape_score
                for k in loss:
                    result.loss[k] += loss[k]
        
        for k in result.loss:
            result.loss_sum[k] = result.loss[k] * result.batch_len
        
        result.mape_sum = result.mape * result.batch_len
        
        epoch_result += result

torch.save({
    'measured': epoch_result.measured,
    'prediction': epoch_result.prediction,
    'instr_len': epoch_result.inst_lens,
    'index': epoch_result.index
}, Path('results', f'{args.model_name}.pkl'))
