import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from utils import recursive_vars
import importlib, inspect

class BasicBlockDataset(Dataset):
    def __init__(self, data, special_tokens, too_long_limit=512, is_training=True):
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

        short_dict = {
            'x': short_x,
            'y': short_y,
            'inst_len': short_inst_len
        }

        long_list = [
            {
                'x': x.unsqueeze(0),
                'y': torch.tensor([y]),
                'inst_len': [inst_len]
            } for x, y, inst_len in zip(long_x, long_y, long_inst_len)
        ]

        return {
            'short': short_dict,
            'long': long_list
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
    def __init__(self, data, special_tokens, too_long_limit=512, using_unroll=False, is_training=True):
        self.pad_idx = special_tokens['PAD']
        self.too_long_limit = too_long_limit
        self.xs = [pad_block(datum.x, self.pad_idx) for datum in data]
        self.ys = [datum.y for datum in data]
        self.inst_lens = [datum.block.num_instrs() for datum in data]
        self.total_size = len(self.xs)
        self.using_unroll = using_unroll
        self.is_training = is_training

    def __len__(self):
        return self.total_size
    
    def __getitem__(self, index):
        if self.using_unroll and self.is_training and self.unrolled[index]:
            return self.xs[index].repeat(2, 1), self.ys[index] * 2, self.inst_lens[index] * 2, True, index
        else:
            return self.xs[index], self.ys[index], self.inst_lens[index], False, index
    
    def update(self, epoch):
        if self.using_unroll and epoch % 3 == 0:
            self.random_unrolling()

    def random_unrolling(self):
        small = torch.tensor(self.inst_lens) < 200
        self.unrolled = torch.rand(self.total_size) < 0.2
        self.unrolled = self.unrolled.logical_and(small)
    
    def collate_fn(self, batch):
        short_x = []
        short_y = []
        long_x = []
        long_y = []
        short_inst_lens = []
        long_inst_lens = []
        short_unrolled = []
        long_unrolled = []

        short_index = []
        long_index = []

        for x, y, inst_len, unrolled, index in batch:
            if x.numel() <= self.too_long_limit:
                short_x.append(x)
                short_y.append(y)
                short_inst_lens.append(inst_len)
                short_unrolled.append(unrolled)
                short_index.append(index)
            else:
                long_x.append(x)
                long_y.append(y)
                long_inst_lens.append(inst_len)
                long_unrolled.append(unrolled)
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
            short_unrolled = torch.tensor(short_unrolled)

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

        if self.using_unroll:
            short_dict['unrolled'] = short_unrolled
            for long_dict, unrolled in zip(long_list, long_unrolled):
                long_dict['unrolled'] = torch.tensor([unrolled])

        return {
            'short': short_dict,
            'long': long_list
        }

class StackedBlockDatasetWithIndex(Dataset):
    def __init__(self, data, special_tokens, too_long_limit=512, is_training=True):
        self.pad_idx = special_tokens['PAD']
        self.too_long_limit = too_long_limit
        self.xs = [pad_block(datum.x, self.pad_idx) for datum in data]
        self.ys = [datum.y for datum in data]
        self.inst_lens = [datum.block.num_instrs() for datum in data]
        self.total_size = len(self.xs)

    def __len__(self):
        return self.total_size
    
    def __getitem__(self, index):
        return self.xs[index], self.ys[index], self.inst_lens[index], index
    
    def collate_fn(self, batch):
        short_x = []
        short_y = []
        long_x = []
        long_y = []
        short_inst_len = []
        long_inst_len = []
        short_index = []
        long_index = []

        for x, y, inst_len, index in batch:
            if x.numel() <= self.too_long_limit:
                short_x.append(x)
                short_y.append(y)
                short_inst_len.append(inst_len)
                short_index.append(index)
            else:
                long_x.append(x)
                long_y.append(y)
                long_inst_len.append(inst_len)
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

        return {
            'short_x': short_x,
            'short_y': short_y,
            'long_x': long_x,
            'long_y': long_y,
            'short_inst_len': short_inst_len,
            'long_inst_len': long_inst_len,
            'short_index': short_index,
            'long_index': long_index,
        }

    
class SimplifyDataset(Dataset):
    def __init__(self, data, special_tokens, too_long_limit=512, is_training=True):
        self.pad_idx = special_tokens['PAD']
        self.too_long_limit = too_long_limit

        self.xs = [torch.tensor(datum.sim) for datum in data]
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
            max_len = max([len(x) for x in short_x])

            short_x = torch.stack([
                F.pad(tensor, (0, max_len-len(tensor)), value=self.pad_idx) for tensor in short_x
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
 

class StackedBlockDatasetTest(Dataset):
    def __init__(self, data, special_tokens, too_long_limit=512, is_training=True):
        self.pad_idx = special_tokens['PAD']
        self.op_idx = special_tokens['OP']
        self.srcs_idx = special_tokens['SRCS']
        self.dsts_idx = special_tokens['DSTS']
        self.start_idx = special_tokens['START']
        self.end_idx = special_tokens['END']
        self.too_long_limit = too_long_limit
        self.xs = [pad_block(datum.x, self.pad_idx) for datum in data]
        self.ys = [datum.y for datum in data]
        self.inst_lens = [datum.block.num_instrs() for datum in data]
        self.total_size = len(self.xs)
        self.is_training = is_training

    def __len__(self):
        return self.total_size
    
    def __getitem__(self, index):
        return self.xs[index], self.ys[index], self.inst_lens[index], index
    
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

        long_list = []
        for x, y, inst_len, index in zip(long_x, long_y, long_inst_lens, long_index):
            x = x.unsqueeze(0)
            long_list.append({
                'x': x,
                'y': torch.tensor([y]),
                'inst_len': [inst_len],
                'index': [index]
            })

        return {
            'short': short_dict,
            'long': long_list
        }

class StackedBertDatasetTest(Dataset):
    def __init__(self, data, special_tokens, too_long_limit=512, is_training=True, vocab_size=700, 
                mask_rate=0.15, update_epoch=3):
        self.pad_idx = special_tokens['PAD']
        self.unk_idx = special_tokens['UNK']
        self.vocab_size = vocab_size
        self.too_long_limit = too_long_limit
        self.xs = [pad_block(datum.x, self.pad_idx) for datum in data]
        self.ys = [datum.y for datum in data]
        self.inst_lens = [datum.block.num_instrs() for datum in data]
        self.total_size = len(self.xs)
        self.is_training = is_training
        self.mask_rate = mask_rate
        self.update_epoch = update_epoch
    
        self.update_random_mask()

    def __len__(self):
        return self.total_size
    
    def __getitem__(self, index):
        return self.xs[index], self.masked_xs[index], self.ys[index], self.inst_lens[index], index
    
    def update(self, epoch_nb):
        if self.update_epoch == -1:
            return
        
        if epoch_nb % self.update_epoch == self.update_epoch - 1:
            self.update_random_mask()

    def update_random_mask(self):
        self.masked_xs = []
        for x in self.xs:
            x_copy = torch.clone(x).detach()
            pad_mask = x_copy == self.pad_idx
            raw = torch.rand_like(x_copy, dtype=torch.float)
            unk_mask = raw < self.mask_rate * 0.8
            rand_mask = (self.mask_rate * 0.8 < raw) & (raw < self.mask_rate * 0.9)

            x_copy[unk_mask] = self.unk_idx

            rand_values = torch.randint_like(x, self.vocab_size)
            x_copy = torch.where(rand_mask, rand_values, x_copy)
            x_copy[pad_mask] = self.pad_idx
            self.masked_xs.append(x_copy)
    
    def collate_fn(self, batch):
        short_x = []
        short_y = []
        long_x = []
        long_y = []
        short_inst_lens = []
        long_inst_lens = []

        short_mask_x = []
        long_mask_x = []

        short_index = []
        long_index = []

        for x, masked_x, y, inst_len, index in batch:
            if x.numel() <= self.too_long_limit:
                short_x.append(x)
                short_y.append(y)
                short_mask_x.append(masked_x)
                short_inst_lens.append(inst_len)
                short_index.append(index)
            else:
                long_x.append(x)
                long_y.append(y)
                long_mask_x.append(masked_x)
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

            short_mask_x = torch.stack([
                F.pad(tensor, 
                    (0, max_shape[1]-tensor.size(1), 0, max_shape[0]-tensor.size(0)), value=self.pad_idx)
                for tensor in short_mask_x
            ])

            short_y = torch.tensor(short_y)
            

        short_dict = {
            'x': short_x,
            'y': short_y,
            'masked_x': short_mask_x,
            'inst_len': short_inst_lens,
            'index': short_index
        }

        long_list = []
        for x, y, long_mask_x, inst_len, index in zip(long_x, long_y, long_mask_x, long_inst_lens, long_index):
            x = x.unsqueeze(0)
            long_mask_x = long_mask_x.unsqueeze(0)
            long_list.append({
                'x': x,
                'y': torch.tensor([y]),
                'masked_x': long_mask_x,
                'inst_len': [inst_len],
                'index': [index]
            })

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

