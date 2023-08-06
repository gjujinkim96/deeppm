import torch
import torch.nn as nn
from .transformer_show_attention import CustomTransformerEncoderLayer, CustomTransformerEncoder
from utils import get_device

from .pos_encoder import get_positional_encoding_1d

class SimpleSum(nn.Module):
    """DeepPM model with Trasformer """
    def __init__(self, cfg, is_show=False):
        super().__init__()

        device = get_device(should_print=False)

    
        block = nn.TransformerEncoderLayer(cfg.dim, cfg.n_heads, device=device, dim_feedforward=cfg.dim_ff,
                                            batch_first=True)
        self.blocks = nn.TransformerEncoder(block, 8, enable_nested_tensor=False)


        self.pad_idx = cfg.pad_idx
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim, padding_idx = cfg.pad_idx,
                                dtype=torch.float32, device=device) # token embedding
        self.pos_embed = get_positional_encoding_1d(cfg.dim)
        self.prediction = nn.Linear(cfg.dim, 1, dtype=torch.float32)
       
    def forward(self, x):
        mask = x == self.pad_idx

        t_output = self.embed(x)
        batch_size, inst_size, seq_size, dim = t_output.shape

        t_output = t_output.view(batch_size * inst_size, seq_size, dim)
        t_output = self.pos_embed(t_output)

        mask = mask.view(batch_size * inst_size, seq_size)

        t_output = t_output.masked_fill(mask.all(dim=-1).unsqueeze(-1).unsqueeze(-1), 1)

        mod_mask = mask.masked_fill(mask.all(dim=-1).unsqueeze(-1), False)

        t_output = self.blocks(t_output, src_key_padding_mask=mod_mask)
        t_output = t_output.masked_fill(mask.all(dim=-1).unsqueeze(-1).unsqueeze(-1), 0)

        t_output = t_output[:,0,:]
        t_output = t_output.view(batch_size, inst_size, dim)
        
        t_output = t_output.sum(dim = 1)
        out = self.prediction(t_output).squeeze(1)

        return out
    