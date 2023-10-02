from tqdm.auto import tqdm

import torch
import torch.autograd as autograd
from torch.utils.data import DataLoader

from operator import itemgetter
from utils import get_device


class BatchResult:
    def __init__(self):
        self.batch_len = 0

        self.measured = []
        self.prediction = []
        self.inst_lens = []
        self.index = []
        
        self.loss_sum = 0

    @property
    def loss(self):
        if self.batch_len == 0:
            return float('nan')
        return self.loss_sum / self.batch_len
        
    def __iadd__(self, other):
        self.batch_len += other.batch_len

        self.measured.extend(other.measured)
        self.prediction.extend(other.prediction)
        self.inst_lens.extend(other.inst_lens)
        self.index.extend(other.index)

        self.loss_sum += other.loss_sum
        return self
    
    def __repr__(self):
        return f'Batch len: {self.batch_len} Loss: {self.loss:.4f}'

def run_model(model, input, is_train=False, loss_fn=None, loss_mod=None, device=None):
    ret = {}
    if device is None:
        device = next(model.parameters()).device

    x = input['x'].to(device)
    output = model(x)
    ret['output'] = output.tolist()

    y = input['y']
    ret['y'] = y.tolist()

    if loss_fn is not None:
        y = y.to(device)
        loss = loss_fn(output, y)

        if loss_mod is not None:
            loss *= loss_mod

        if is_train:
            loss.backward()

        ret['loss'] = loss.item()
    
    return ret

def run_batch(batch, model, is_train=False, loss_fn=None, device=None):
    short, long = itemgetter('short', 'long')(batch)

    batch_result = BatchResult()
    short_len = len(short['y'])
    long_len = len(long)
    batch_len = short_len + long_len
    batch_result.batch_len = batch_len
    batch_result.inst_lens = short['inst_len'] + [item['inst_len'][0] for item in long]
    batch_result.index = short['index'] + [item['index'][0] for item in long]
    
    if short_len > 0:
        loss_mod = short_len / batch_len if long_len > 0 else None

        model_result = run_model(model, short, is_train=is_train, loss_fn=loss_fn, loss_mod=loss_mod, device=device)
        batch_result.measured.extend(model_result['y'])
        batch_result.prediction.extend(model_result['output'])
        batch_result.loss_sum += model_result['loss'] * batch_len
    
    if long_len > 0:
        for long_item in long:
            model_result = run_model(model, long_item, is_train=is_train, loss_fn=loss_fn, loss_mod=1/batch_len, device=device)
            batch_result.measured.extend(model_result['y'])
            batch_result.prediction.extend(model_result['output'])
            batch_result.loss_sum += model_result['loss'] * batch_len
    return batch_result

def validate(model, ds, loss_fn=None, device=None, batch_size=8):
    if device is None:
        device = get_device(False)

    model.eval()
    model.to(device)

    loader = DataLoader(ds, shuffle=False, batch_size=batch_size, collate_fn=ds.collate_fn)
    epoch_result = BatchResult()

    with torch.no_grad():
        for batch in tqdm(loader):
            epoch_result += run_batch(batch, model, is_train=False, loss_fn=loss_fn, device=device)

    return epoch_result
