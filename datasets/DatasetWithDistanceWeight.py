import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from .utils import pad_block, TorchDict, collate_function
   
def make_attention_weight(mask, is_continual_pad=True):
    sizes = (~mask).sum(dim=1)
    maximum_size = mask.size(1)

    all_masking = []
    
    for idx, s in enumerate(sizes):
        cur_mask = ~mask[idx]
        
        i, j = torch.meshgrid(
            torch.arange(s, device=mask.device), torch.arange(s, device=mask.device), indexing='ij'
        )
    
        
        if is_continual_pad:
            masking = F.pad((s - abs(i - j)) / s, (0, maximum_size-s, 0, maximum_size-s), value=0)
        else:
            tmp = torch.full((maximum_size, maximum_size), 0.0, device=mask.device)
            tmp[cur_mask] = F.pad((s-abs(i - j)) / s, (0, maximum_size-s), value=0)
        
            masking = torch.full((maximum_size, maximum_size), 0.0, device=mask.device)
            masking[:, cur_mask] = tmp[:, :s]

        all_masking.append(masking)

    all_masking = torch.stack(all_masking)
    
    return all_masking

class DatasetWithDistanceWeight(Dataset):
    def __init__(self, data, special_tokens, too_long_limit=512,
                return_bb_mask=True, return_seq_mask=True, return_op_mask=True):
        self.pad_idx = special_tokens['PAD']
        self.too_long_limit = too_long_limit
        self.xs = [pad_block(datum.x, self.pad_idx) for datum in data]
        self.ys = [datum.y for datum in data]
        self.inst_lens = [datum.num_instrs for datum in data]
        self.total_size = len(self.xs)
        
        self.return_bb_mask = return_bb_mask
        self.return_seq_mask = return_seq_mask
        self.return_op_mask = return_op_mask
        self.code_id = [datum.code_id for datum in data]

    def __len__(self):
        return self.total_size
    
    def __getitem__(self, index):
        return self.xs[index], self.ys[index], self.inst_lens[index], self.code_id[index]
    
    def make_input(self, x):
        x_dict = {
            'x': x,
        }

        batch_size, inst_size, seq_size = x.shape
        mask = x == self.pad_idx
        bb_mask = mask.view(batch_size, inst_size * seq_size)
        seq_mask = mask.view(batch_size * inst_size, seq_size)
        op_mask = mask.all(dim=2)
        if self.return_bb_mask:
            bb_attn_mod = make_attention_weight(bb_mask, is_continual_pad=False)
            x_dict['bb_attn_mod'] = bb_attn_mod

        if self.return_seq_mask:
            seq_attn_mod = make_attention_weight(seq_mask)
            x_dict['seq_attn_mod'] = seq_attn_mod

        if self.return_op_mask:
            op_attn_mod = make_attention_weight(op_mask)
            x_dict['op_attn_mod'] = op_attn_mod
        
        return TorchDict(**x_dict)

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
        short_x = self.make_input(short_x)
        return short_x, short_y
    
    def collate_long(self, long_x, long_y, long_inst_lens, long_index):
        return [
            {
                'x': self.make_input(x.unsqueeze(0)),
                'y': torch.tensor([y]),
                'inst_len': [inst_len],
                'index': [index]
            } for x, y, inst_len, index in zip(long_x, long_y, long_inst_lens, long_index)
        ]

    def collate_fn(self, batch):
        return collate_function(batch, self)