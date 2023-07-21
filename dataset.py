import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

class BasicBlockDataset(Dataset):
    def __init__(self, embeddings, max_len, pad_idx=0):
        self.embeddings = embeddings
        self.pad_idx = pad_idx
        self.max_len = max_len

    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, index):
        return self.embeddings[index]
    
    def block_collate_fn(self, batch):
        tensors = [torch.tensor(item.x) for item in batch]
        max_len = max([len(item) for item in tensors])
        xs = torch.stack(
            [
                F.pad(item, (0, max_len-len(item)), value=self.pad_idx)
                    for item in tensors
            ]
        )
        raw_ys = torch.tensor([item.y for item in batch])
        ys = torch.log(raw_ys + 1e-4)
        return xs[:, :self.max_len], ys, raw_ys
    
if __name__ == '__main__':
    from data.data_cost import load_dataset

    file_name = 'training_data/intel_core.data'
    data = load_dataset(file_name, small_size=True)
    train_ds = BasicBlockDataset(data.train, data.pad_idx)
    test_ds = BasicBlockDataset(data.test, data.pad_idx)

    loader = DataLoader(train_ds, batch_size=8, collate_fn=train_ds.block_collate_fn)
    for sample in loader:
        print(sample)
        print(sample[0].size())
        break
