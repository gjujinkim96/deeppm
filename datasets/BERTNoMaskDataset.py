import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

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

