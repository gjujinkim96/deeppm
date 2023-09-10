import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from utils import recursive_vars
import importlib, inspect    

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
    def __init__(self, data, special_tokens, too_long_limit=512, is_training=True):
        self.pad_idx = special_tokens['PAD']
        self.too_long_limit = too_long_limit
        self.xs = [pad_block(datum.x, self.pad_idx) for datum in data]
        self.ys = [datum.y for datum in data]
        self.inst_lens = [datum.block.num_instrs() for datum in data]
        self.total_size = len(self.xs)
        self.is_training = is_training

    def __len__(self):
        return self.total_size
    
    def __getitem__(self, index):
        return self.xs[index], self.ys[index], self.inst_lens[index], False, index

    def collate_fn(self, batch):
        short_x = []
        short_y = []
        long_x = []
        long_y = []
        short_inst_lens = []
        long_inst_lens = []

        short_index = []
        long_index = []

        for x, y, inst_len, unrolled, index in batch:
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
 

class BERTNoMaskDataset(Dataset):
    def __init__(self, data, special_tokens, too_long_limit=512, is_training=True):
        self.too_long_limit = too_long_limit
        self.xs = [torch.cat([torch.tensor(tmp) for tmp in datum.x]) for datum in data]
        self.data = data
        self.total_len = len(self.data)
        self.pad_idx = special_tokens['PAD']

    def __len__(self):
        return self.total_len

    def __getitem__(self, index):
        cur = self.data[index]
        return self.xs[index], cur.y, cur.block.num_instrs(), index

    def collate_fn(self, batch):
        short_max_len = 0
        short_x = []
        short_y = []
        long_x = []
        long_y = []

        short_inst_len = []
        long_inst_len = []

        short_index = []
        long_index = []

        for idx, (x, y, num_instrs, index) in enumerate(batch):
            if len(x) <= self.too_long_limit:
                short_x.append(x)
                short_y.append(y)
                short_inst_len.append(num_instrs)
                short_index.append(index)
            else:
                long_x.append(x)
                long_y.append(y)
                long_inst_len.append(num_instrs)
                long_index.append(index)

        if len(short_x) > 0:
            short_max_len = max([len(x) for x in short_x])
            short_x = torch.stack(
                [
                    F.pad(item, (0, short_max_len-len(item)), value=self.pad_idx)
                        for item in short_x
                ]
            )

            short_y = torch.tensor(short_y)

        short_dict = {
            'x': short_x,
            'y': short_y,
            'inst_len': short_inst_len,
            'index': short_index,
        }

        long_list = [
            {
                'x': x.unsqueeze(0),
                'y': torch.tensor([y]),
                'inst_len': [inst_len],
                'index': [index]
            } for x, y, inst_len, index in zip(long_x, long_y, long_inst_len, long_index)
        ]

        return {
            'short': short_dict,
            'long': long_list
        }

class_dict = {}

module = importlib.import_module('dataset')
for name, cls in inspect.getmembers(module, inspect.isclass):
    if cls.__module__ == module.__name__:
        class_dict[name] = cls 

def load_dataset_from_cfg(data, cfg, show=False):
    train_dataset, val_dataset, test_dataset = load_dataset(data, cfg.data.dataset_class, recursive_vars(cfg.data.dataset_setting),
                        recursive_vars(cfg.data.special_token_idx))
    if show:
        print(f'Train Dataset: {len(train_dataset)}  Val Dataset: {len(val_dataset)}  Test Dataset: {len(test_dataset)}')
    return train_dataset, val_dataset, test_dataset
    

def load_dataset(data, dataset_type, dataset_setting={}, special_tokens={}):
    if dataset_type not in class_dict:
        raise NotImplementedError()
    
    dataset_class = class_dict[dataset_type]
    dataset_setting['is_training'] = True
    train_dataset = dataset_class(data.train, special_tokens=special_tokens, **dataset_setting)

    dataset_setting['is_training'] = False
    val_dataset = dataset_class(data.val, special_tokens=special_tokens, **dataset_setting)
    test_dataset = dataset_class(data.test, special_tokens=special_tokens, **dataset_setting)

    return train_dataset, val_dataset, test_dataset

