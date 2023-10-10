import torch
import torch.nn.functional as F

class TorchDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def to(self, device):
        for k, v in self.items():
            if hasattr(v, 'to'):
                self[k] = v.to(device)
        return self

def pad_block(item, pad_idx):
    bb_list = []
    token_len = 0

    tensors = [torch.tensor(inputs) for inputs in item]
    max_len = max(map(len, tensors))
    
    return torch.stack([
        F.pad(tensor, (0, max_len-len(tensor)), value=pad_idx)
            for tensor in tensors
    ])

def collate_function(batch, dataset):
    short_x = []
    short_y = []
    long_x = []
    long_y = []
    short_inst_lens = []
    long_inst_lens = []

    short_index = []
    long_index = []

    for x, y, inst_len, index in batch:
        if x.numel() <= dataset.too_long_limit:
            short_x.append(x)
            short_y.append(y)
            short_inst_lens.append(inst_len)
            short_index.append(index)
        else:
            long_x.append(x)
            long_y.append(y)
            long_inst_lens.append(inst_len)
            long_index.append(index)

    if len(short_x) > 0:
        short_x, short_y = dataset.collate_short(short_x, short_y)

    short_dict = {
        'x': short_x,
        'y': short_y,
        'inst_len': short_inst_lens,
        'index': short_index
    }

    long_list = dataset.collate_long(long_x, long_y, long_inst_lens, long_index)

    return {
        'short': short_dict,
        'long': long_list
    }