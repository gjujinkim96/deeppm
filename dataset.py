import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from utils import recursive_vars
import importlib, inspect

# def load_dataset(data, cfg):
#     if cfg.data.stacked:
#         train_ds = StackedBlockDataset(data.train, cfg.model)
#         test_ds = StackedBlockDataset(data.test, cfg.model)
#     else:
#         train_ds = BasicBlockDataset(data.train, cfg.model)
#         test_ds = BasicBlockDataset(data.test, cfg.model)
        
#     print(f'Train Dataset: {len(train_ds)}  Test Dataset: {len(test_ds)}')
#     return train_ds, test_ds

class BasicBlockDataset(Dataset):
    def __init__(self, data, special_tokens, too_long_limit=512):
        self.embeddings = data
        self.pad_idx = special_tokens['PAD']
        self.too_long_limit = too_long_limit

    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, index):
        return self.embeddings[index]
    
    def collate_fn(self, batch):
        short_max_len = 0
        short_x = []
        short_y = []
        long_x = []
        long_y = []

        short_inst_len = []
        long_inst_len = []

        for idx, item in enumerate(batch):
            ten = torch.tensor(item.x)
            if len(ten) <= self.too_long_limit:
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
    def __init__(self, data, special_tokens, too_long_limit=512):
        self.pad_idx = special_tokens['PAD']
        self.too_long_limit = too_long_limit
        self.xs = [pad_block(datum.x, self.pad_idx) for datum in data]
        self.ys = [datum.y for datum in data]
        self.inst_lens = [datum.block.num_instrs() for datum in data]
        self.total_size = len(self.xs)

    def __len__(self):
        return self.total_size
    
    def __getitem__(self, index):
        return self.xs[index], self.ys[index], self.inst_lens[index]
    
    def collate_fn(self, batch):
        short_x = []
        short_y = []
        long_x = []
        long_y = []
        short_inst_len = []
        long_inst_len = []

        for x, y, inst_len in batch:
            if x.numel() <= self.too_long_limit:
                short_x.append(x)
                short_y.append(y)
                short_inst_len.append(inst_len)
            else:
                long_x.append(x)
                long_y.append(y)
                long_inst_len.append(inst_len)

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

        return {
            'short_x': short_x,
            'short_y': short_y,
            'long_x': long_x,
            'long_y': long_y,
            'short_inst_len': short_inst_len,
            'long_inst_len': long_inst_len,
        }
    

class_dict = {}

module = importlib.import_module('dataset')
for name, cls in inspect.getmembers(module, inspect.isclass):
    if cls.__module__ == module.__name__:
        class_dict[name] = cls 

def load_dataset_from_cfg(data, cfg, show=False):
    train_dataset, test_dataset = load_dataset(data, cfg.data.dataset_class, recursive_vars(cfg.data.dataset_setting),
                        recursive_vars(cfg.data.special_token_idx))
    if show:
        print(f'Train Dataset: {len(train_dataset)}  Test Dataset: {len(test_dataset)}')
    return train_dataset, test_dataset
    

def load_dataset(data, dataset_type, dataset_setting={}, special_tokens={}):
    if dataset_type not in class_dict:
        raise NotImplementedError()
    
    dataset_class = class_dict[dataset_type]
    train_dataset = dataset_class(data.train, special_tokens=special_tokens, **dataset_setting)
    test_dataset = dataset_class(data.test, special_tokens=special_tokens, **dataset_setting)

    return train_dataset, test_dataset

