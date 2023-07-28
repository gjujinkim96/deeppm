import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

def load_dataset(data, train_cfg, model_cfg):
    if train_cfg.stacked:
        train_ds = StackedBlockDataset(data.train, model_cfg)
        test_ds = StackedBlockDataset(data.train, model_cfg)
    else:
        train_ds = BasicBlockDataset(data.train, model_cfg)
        test_ds = BasicBlockDataset(data.test, model_cfg)
    return train_ds, test_ds

class BasicBlockDataset(Dataset):
    def __init__(self, embeddings, model_cfg):
        self.embeddings = embeddings
        self.pad_idx = model_cfg.pad_idx
        self.max_len = model_cfg.max_len

    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, index):
        return self.embeddings[index]
    
    def block_collate_fn(self, batch):
        short_max_len = 0
        short_x = []
        short_y = []
        long_x = []
        long_y = []

        short_inst_len = []
        long_inst_len = []

        for idx, item in enumerate(batch):
            ten = torch.tensor(item.x)
            if len(ten) <= self.max_len:
                short_max_len = max(len(ten), short_max_len)
                short_x.append(ten)
                short_y.append(item.y)
                short_inst_len.append(item.block.num_instrs())
            else:
                long_x.append(ten)
                long_y.append(item.y)
                long_inst_len.append(item.block.num_instrs())

        if len(short_x) > 0:
            short_x = torch.stack(
                [
                    F.pad(item, (0, short_max_len-len(item)), value=self.pad_idx)
                        for item in short_x
                ]
            )
            short_y = torch.tensor(short_y)

        return {
            'short_x': short_x,
            'short_y': short_y,
            'long_x': long_x,
            'long_y': long_y,
            'short_inst_len': short_inst_len,
            'long_inst_len': long_inst_len,   
        }

def pad_block(item, pad_idx):
    bb_list = []
    token_len = 0

    tensors = [torch.tensor(inputs) for inputs in item]
    max_len = max(map(len, tensors))
    
    return torch.stack([
        F.pad(tensor, (0, max_len-len(tensor)), value=pad_idx)
            for tensor in tensors
    ])


class StackedBlockDataset(Dataset):
    def __init__(self, data, model_cfg):
        self.pad_idx = model_cfg.pad_idx
        self.short_len = model_cfg.max_len
        self.xs = [pad_block(datum.x, self.pad_idx) for datum in data]
        self.ys = [datum.y for datum in data]
        self.inst_lens = [datum.block.num_instrs() for datum in data]
        self.total_size = len(self.xs)

    def __len__(self):
        return self.total_size
    
    def __getitem__(self, index):
        return self.xs[index], self.ys[index], self.inst_lens[index]
    
    def block_collate_fn(self, batch):
        short_x = []
        short_y = []
        long_x = []
        long_y = []
        short_inst_len = []
        long_inst_len = []

        for x, y, inst_len in batch:
            if x.numel() <= self.short_len:
                short_x.append(x)
                short_y.append(y)
                short_inst_len.append(inst_len)
            else:
                long_x.append(x)
                long_y.append(y)
                long_inst_len.append(inst_len)

        max_shape = torch.stack(
                [torch.tensor(x.shape) for x in short_x]
            ).max(dim=0)[0]
        
        short_x = torch.stack([
            F.pad(tensor, 
                  (0, max_shape[1]-tensor.size(1), 0, max_shape[0]-tensor.size(0)), value=self.pad_idx)
            for tensor in short_x
        ])

        short_y = torch.tensor(short_y)

        return {
            'short_x': short_x,
            'short_y': short_y,
            'long_x': long_x,
            'long_y': long_y,
            'short_inst_len': short_inst_len,
            'long_inst_len': long_inst_len,
        }
    