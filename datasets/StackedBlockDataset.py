import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from .utils import pad_block, collate_function

class StackedBlockDataset(Dataset):
    def __init__(self, data, special_tokens, too_long_limit=512):
        self.pad_idx = special_tokens['PAD']
        self.too_long_limit = too_long_limit
        self.xs = [pad_block(datum.x, self.pad_idx) for datum in data]
        self.ys = [datum.y for datum in data]
        self.inst_lens = [datum.block.num_instrs() for datum in data]
        self.total_size = len(self.xs)
        self.code_id = [datum.code_id for datum in data]

    def __len__(self):
        return self.total_size
    
    def __getitem__(self, index): 
        return self.xs[index], self.ys[index], self.inst_lens[index], self.code_id[index]

    def collate_short(self, short_x, short_y):
        max_shape = torch.stack(
                [torch.tensor(x.shape) for x in short_x]
            ).max(dim=0)[0]
        
        short_x = torch.stack([
            F.pad(tensor, 
                (0, max_shape[1]-tensor.size(1), 0, max_shape[0]-tensor.size(0)), value=self.pad_idx)
            for tensor in short_x
        ])

        short_y = torch.tensor(short_y)
        return short_x, short_y
    
    def collate_long(self, long_x, long_y, long_inst_lens, long_index):
        return [
            {
                'x': x.unsqueeze(0),
                'y': torch.tensor([y]),
                'inst_len': [inst_len],
                'index': [index]
            } for x, y, inst_len, index in zip(long_x, long_y, long_inst_lens, long_index)
        ]
    
    def collate_fn(self, batch):
        return collate_function(batch, self)
    