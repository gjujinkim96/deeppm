import torch
import torch.nn.functional as F

class TorchDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def to(self, device):
        for k, v in self.items():
            if hasattr(v, 'to'):
                self[k] = v.to(device)
        return self

def pad_block(item, pad_idx):
    bb_list = []
    token_len = 0

    tensors = [torch.tensor(inputs) for inputs in item]
    max_len = max(map(len, tensors))
    
    return torch.stack([
        F.pad(tensor, (0, max_len-len(tensor)), value=pad_idx)
            for tensor in tensors
    ])