import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from .utils import pad_block, TorchDict


def make_attention_weight_raw(mask, length_limit=40, offset_by_zero_idx=True, is_continual_pad=True):
    sizes = (~mask).sum(dim=1)
    maximum_size = mask.size(1)

    pad_idx = length_limit+1
    all_masking = []
    
    for idx, s in enumerate(sizes):
        cur_mask = ~mask[idx]
        
        i, j = torch.meshgrid(
            torch.arange(s, device=mask.device), torch.arange(s, device=mask.device), indexing='ij'
        )

        if is_continual_pad:
            masking = F.pad(j - i, (0, maximum_size-s, 0, maximum_size-s), value=pad_idx)
        else:
            tmp = torch.full((maximum_size, maximum_size), pad_idx, device=mask.device)
            tmp[cur_mask] = F.pad(j - i, (0, maximum_size-s), value=pad_idx)
        
            masking = torch.full((maximum_size, maximum_size), pad_idx, device=mask.device)
            masking[:, cur_mask] = tmp[:, :s]

    
        
        all_masking.append(masking)

    all_masking = torch.stack(all_masking)
    pad_pos = all_masking == pad_idx
    
    all_masking[all_masking >= length_limit] = length_limit
    all_masking[all_masking <= -length_limit] = -length_limit
    all_masking[pad_pos] = pad_idx

    if offset_by_zero_idx:
        all_masking += length_limit
    return all_masking, sizes

0 
   
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
    def __init__(self, data, special_tokens, too_long_limit=512, dist_weight_len_limit=40, 
                return_raw_dist_weight=False, offset_by_zero_idx=True, is_training=True,
                return_bb_mask=True, return_seq_mask=True, return_op_mask=True):
        self.pad_idx = special_tokens['PAD']
        self.too_long_limit = too_long_limit
        self.xs = [pad_block(datum.x, self.pad_idx) for datum in data]
        self.ys = [datum.y for datum in data]
        self.inst_lens = [datum.block.num_instrs() for datum in data]
        self.total_size = len(self.xs)
        self.is_training = is_training
        self.dist_weight_len_limit = dist_weight_len_limit
        self.return_raw_dist_weight = return_raw_dist_weight
        self.offset_by_zero_idx = offset_by_zero_idx
        self.return_bb_mask = return_bb_mask
        self.return_seq_mask = return_seq_mask
        self.return_op_mask = return_op_mask

    def __len__(self):
        return self.total_size
    
    def __getitem__(self, index):
        return self.xs[index], self.ys[index], self.inst_lens[index], index
    
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
            if self.return_raw_dist_weight:
                bb_attn_mod, bb_sizes = make_attention_weight_raw(bb_mask, length_limit=self.dist_weight_len_limit,
                                            offset_by_zero_idx=self.offset_by_zero_idx, is_continual_pad=False)
                x_dict['bb_attn_mod'] = bb_attn_mod
                x_dict['bb_sizes'] = bb_sizes
            else:
                bb_attn_mod = make_attention_weight(bb_mask, is_continual_pad=False)
                x_dict['bb_attn_mod'] = bb_attn_mod

        if self.return_seq_mask:
            if self.return_raw_dist_weight:
                seq_attn_mod, seq_sizes = make_attention_weight_raw(seq_mask, length_limit=self.dist_weight_len_limit,
                                            offset_by_zero_idx=self.offset_by_zero_idx)
                x_dict['seq_attn_mod'] = seq_attn_mod
                x_dict['seq_sizes'] = seq_sizes
            else:
                seq_attn_mod = make_attention_weight(seq_mask)
                x_dict['seq_attn_mod'] = seq_attn_mod

        if self.return_op_mask:
            if self.return_raw_dist_weight:
                op_attn_mod, op_sizes = make_attention_weight_raw(op_mask, length_limit=self.dist_weight_len_limit,
                                            offset_by_zero_idx=self.offset_by_zero_idx)
                x_dict['op_attn_mod'] = op_attn_mod
                x_dict['op_sizes'] = op_sizes
            else:
                op_attn_mod = make_attention_weight(op_mask)
                x_dict['op_attn_mod'] = op_attn_mod
        
        return TorchDict(**x_dict)

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
            short_x = self.make_input(short_x)

        short_dict = {
            'x': short_x,
            'y': short_y,
            'inst_len': short_inst_lens,
            'index': short_index
        }

        long_list = []
        for x, y, inst_len, index in zip(long_x, long_y, long_inst_lens, long_index):
            cur_x = x.unsqueeze(0)
            cur_x = self.make_input(cur_x)

            cur_dict = {
                'x': cur_x,
                'y': torch.tensor([y]),
                'inst_len': [inst_len],
                'index': [index]
            }
            long_list.append(cur_dict)
        
        return {
            'short': short_dict,
            'long': long_list
        }
 
