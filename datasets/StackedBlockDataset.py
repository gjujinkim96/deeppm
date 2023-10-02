import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from .utils import pad_block

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
    
    # TODO: change index to code_id
    def __getitem__(self, index): 
        return self.xs[index], self.ys[index], self.inst_lens[index], self.code_id[index]

    def collate_fn(self, batch):
        short_x = []
        short_y = []
        long_x = []
        long_y = []
        short_inst_lens = []
        long_inst_lens = []

        short_index = []
        long_index = []

        for x, y, inst_len, index in batch:
            if x.numel() <= self.too_long_limit:
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
            max_shape = torch.stack(
                    [torch.tensor(x.shape) for x in short_x]
                ).max(dim=0)[0]
            
            short_x = torch.stack([
                F.pad(tensor, 
                    (0, max_shape[1]-tensor.size(1), 0, max_shape[0]-tensor.size(0)), value=self.pad_idx)
                for tensor in short_x
            ])

            short_y = torch.tensor(short_y)

        short_dict = {
            'x': short_x,
            'y': short_y,
            'inst_len': short_inst_lens,
            'index': short_index
        }

        long_list = [
            {
                'x': x.unsqueeze(0),
                'y': torch.tensor([y]),
                'inst_len': [inst_len],
                'index': [index]
            } for x, y, inst_len, index in zip(long_x, long_y, long_inst_lens, long_index)
        ]

        return {
            'short': short_dict,
            'long': long_list
        }