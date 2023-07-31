import torch
import torch.nn as nn
from utils import get_device

from .pos_encoder import get_positional_encoding_1d

from torch.utils.checkpoint import checkpoint


class StackedDeepPM800(nn.Module):
    """DeepPM model with Trasformer """
    def __init__(self, cfg):
        super().__init__()

        device = get_device(should_print=False)
        block = nn.TransformerEncoderLayer(cfg.dim, cfg.n_heads, device=device, dim_feedforward=cfg.dim_ff,
                                            batch_first=True)
        self.f_block = nn.TransformerEncoder(block, 3, enable_nested_tensor=False)
        block = nn.TransformerEncoderLayer(cfg.dim, cfg.n_heads, device=device, dim_feedforward=cfg.dim_ff,
                                            batch_first=True)
        self.s_block = nn.TransformerEncoder(block, 3, enable_nested_tensor=False)
        block = nn.TransformerEncoderLayer(cfg.dim, cfg.n_heads, device=device, dim_feedforward=cfg.dim_ff,
                                            batch_first=True)
        self.t_block = nn.TransformerEncoder(block, 2, enable_nested_tensor=False)

        self.pad_idx = cfg.pad_idx
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim, padding_idx = cfg.pad_idx,
                                dtype=torch.float32, device=device) # token embedding
        self.pos_embed = get_positional_encoding_1d(cfg.dim)
        self.prediction = nn.Sequential(
            nn.Dropout(cfg.pred_drop),
            nn.Linear(cfg.dim, 1, dtype=torch.float32)
        )

        self.dummy = torch.zeros(1, requires_grad=True)
    
    def checkpoint_forward(self, x):
        output = checkpoint(self._wrapper_forward_1, x, self.dummy)
        output = checkpoint(self._wrapper_forward_2, output, self.dummy)
        return self._forward_3(output)
       
    def forward(self, x):
        t_outputs = self._forward_1(x)
        t_outputs = self._forward_2(t_outputs)
        return self._forward_3(t_outputs)
    
    def _wrapper_forward_1(self, x, dummy):
        return self._forward_1(x)
    
    def _wrapper_forward_2(self, x, dummy):
        return self._forward_2(x)
    
    def _forward_1(self, x):
        mask = x == self.pad_idx

        t_output = self.embed(x)
        batch_size, inst_size, seq_size, dim = t_output.shape

        t_output = t_output.view(batch_size, inst_size*seq_size, dim)
        t_output = self.pos_embed(t_output)

        mask = mask.view(batch_size, inst_size * seq_size)

        t_output = self.f_block(t_output, src_key_padding_mask=mask)
        return t_output, mask
    
    def _forward_2(self, given):
        x, mask = given
        t_output = self.s_block(x, src_key_padding_mask=mask)
        return t_output, mask
    
    def _forward_3(self, given):
        x, mask = given
        t_output = self.t_block(x, src_key_padding_mask=mask)
        t_output = t_output.masked_fill(mask.unsqueeze(-1), 0)

        t_output = t_output.sum(dim=1)
        out = self.prediction(t_output).squeeze(1)
        return out
