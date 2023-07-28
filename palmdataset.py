import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from tqdm.auto import tqdm

def parse_line(line):
    ops = '[]+-*'
    for op in ops:
        line = line.replace(op, f' {op} ')
    line = line.replace(',', ' ')
    final = line.split()
    return ' '.join(final)
   

class PalmDataset(Dataset):
    def __init__(self, data, palmtree):
        self.data = []
        for item in tqdm(data):
            y = item.y
            raw = [parse_line(x.intel) for x in item.block.instrs]
            encoded = palmtree.encode(raw)
            self.data.append({
                'x': torch.tensor(encoded),
                'raw': raw,
                'y': y
            })

        self.pad_idx = palmtree.vocab.pad_index
        
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]
    
    def block_collate_fn(self, batch):
        xs = [item['x'] for item in batch]
        max_len = max([len(item) for item in xs])
        masks = torch.stack([
                    torch.concat([torch.zeros(len(item)), torch.ones(max_len-len(item))]) 
                        for item in xs
                ])
        x = torch.stack(
            [
                F.pad(item, (0, 0, 0, max_len-len(item)), value=self.pad_idx)
                    for item in xs
            ]
        )
        y = torch.tensor([item['y'] for item in batch])
        
        
        return x, y, masks
