import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BasicBlockDataset(Dataset):
    def __init__(self, embeddings):
        self.embeddings = embeddings

    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, index):
        return self.embeddings[index]
    
def block_collate_fn(batch):
    max_size = torch.tensor([item.x.size() for item in batch]).max(dim=0)[0]
    xs = torch.stack([
        F.pad(item.x,
                (0, max_size[1]-item.x.size(1), 0, max_size[0]-item.x.size(0))) 
        for item in batch
    ])
    ys = torch.tensor([item.y for item in batch])
    return xs, ys
    
if __name__ == '__main__':
    from data.data_cost import DataInstructionEmbedding

    file_name = 'training_data/intel_core.data'
    data = DataInstructionEmbedding()

    data.raw_data = torch.load(file_name)[:1000]

    data.read_meta_data()
    data.prepare_data()
    data.generate_datasets()

    train_ds = BasicBlockDataset(data.train)
    test_ds = BasicBlockDataset(data.test)

    loader = DataLoader(train_ds, batch_size=8, collate_fn=block_collate_fn)
    for sample in loader:
        print(sample)
        print(sample[0].size())
        break
