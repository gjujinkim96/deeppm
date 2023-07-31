import torch
import torch.nn as nn
from utils import get_device

from .pos_encoder import get_positional_encoding_1d, get_positional_encoding_2d

class Pos2dInst(nn.Module):
    """DeepPM model with Trasformer """
    def __init__(self, cfg):
        super().__init__()


        device = get_device(should_print=False)

        block = nn.TransformerEncoderLayer(cfg.dim, cfg.n_heads, device=device, dim_feedforward=cfg.dim_ff,
                                            batch_first=True)
        self.f_block = nn.TransformerEncoder(block, 2, enable_nested_tensor=False)

        block = nn.TransformerEncoderLayer(cfg.dim, cfg.n_heads, device=device, dim_feedforward=cfg.dim_ff,
                                            batch_first=True)
        self.s_block = nn.TransformerEncoder(block, 2, enable_nested_tensor=False)

        block = nn.TransformerEncoderLayer(cfg.dim, cfg.n_heads, device=device, dim_feedforward=cfg.dim_ff,
                                            batch_first=True)
        self.t_block = nn.TransformerEncoder(block, 4, enable_nested_tensor=False)


        self.pad_idx = cfg.pad_idx
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim, padding_idx = cfg.pad_idx,
                                dtype=torch.float32, device=device) # token embedding
        self.pos2d_embed = get_positional_encoding_2d(cfg.dim)
        self.pos1d_embed = get_positional_encoding_1d(cfg.dim)

        self.prediction = nn.Sequential(
            nn.Dropout(cfg.pred_drop),
            nn.Linear(cfg.dim, 1, dtype=torch.float32)
        )
       
    def forward(self, x, debug=False):
        mask = x == self.pad_idx

        t_output = self.embed(x)
        batch_size, inst_size, seq_size, dim = t_output.shape

        t_output = self.pos2d_embed(t_output)

        t_output = t_output.view(batch_size, inst_size * seq_size, dim)
        mask = mask.view(batch_size, inst_size * seq_size)

        t_output = self.f_block(t_output, src_key_padding_mask=mask)
        t_output = t_output.view(batch_size * inst_size, seq_size, dim)
        mask = mask.view(batch_size * inst_size, seq_size)

        t_output = t_output.masked_fill(mask.all(dim=-1).unsqueeze(-1).unsqueeze(-1), 1)

        mod_mask = mask.masked_fill(mask.all(dim=-1).unsqueeze(-1), False)

        t_output = self.s_block(t_output, src_key_padding_mask=mod_mask)
        

        t_output = t_output.masked_fill(mask.all(dim=-1).unsqueeze(-1).unsqueeze(-1), 0)

        t_output = t_output[:,0,:]
        t_output = t_output.view(batch_size, inst_size, dim)
        t_output = self.pos1d_embed(t_output)

        mask = mask.view(batch_size, inst_size, seq_size)
        mask = mask.all(dim=-1)

        t_output = self.t_block(t_output, src_key_padding_mask=mask)
        t_output = t_output.sum(dim = 1)
        out = self.prediction(t_output).squeeze(1)

        return out
    