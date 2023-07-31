import torch
import torch.nn as nn
from utils import get_device

from .custom_transformer import PositionalEncoding

class DeepPM(nn.Module):
    """DeepPM model with Trasformer """
    def __init__(self, cfg):
        super().__init__()

        device = get_device(should_print=False)
        block = nn.TransformerEncoderLayer(cfg.dim, cfg.n_heads, device=device,
                                           dim_feedforward=cfg.dim_ff, batch_first=True)
        self.blocks = nn.TransformerEncoder(block, cfg.n_layers, enable_nested_tensor=False)

        self.pad_idx = cfg.pad_idx
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim, padding_idx = cfg.pad_idx,
                                dtype=torch.float32, device=device) # token embedding
        
        self.pos_embed = PositionalEncoding(cfg.dim, 4000).to(get_device())
        self.prediction = nn.Sequential(
            nn.Dropout(),
            nn.Linear(cfg.dim, 1, dtype=torch.float32)
        )
       
    def forward(self, item):
        padding_mask = (item == self.pad_idx)
        t_output = self.embed(item)
        t_output = self.pos_embed(t_output)

        # batch_size, n_instr, n_dim = t_output.size()

        # B, L, D
        t_output = self.blocks(t_output, src_key_padding_mask=padding_mask)

        t_output = t_output[:, 0, :]
        out = self.prediction(t_output).squeeze(1)

        return out
