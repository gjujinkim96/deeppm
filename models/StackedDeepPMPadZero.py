import torch
import torch.nn as nn
from .transformer_show_attention import CustomTransformerEncoderLayer, CustomTransformerEncoder
from utils import get_device

from .pos_encoder import get_positional_encoding_1d

class StackedDeepPMPadZero(nn.Module):
    """DeepPM model with Trasformer """
    def __init__(self, cfg, is_show=False):
        super().__init__()

        device = get_device(should_print=False)

        self.is_show = is_show
        if self.is_show:
            block = CustomTransformerEncoderLayer(cfg.dim, cfg.n_heads, device=device, dim_feedforward=cfg.dim_ff,
                                                batch_first=True)
            self.f_block = CustomTransformerEncoder(block, 2, enable_nested_tensor=False)

            block = CustomTransformerEncoderLayer(cfg.dim, cfg.n_heads, device=device, dim_feedforward=cfg.dim_ff,
                                                batch_first=True)
            self.s_block = CustomTransformerEncoder(block, 2, enable_nested_tensor=False)

            block = CustomTransformerEncoderLayer(cfg.dim, cfg.n_heads, device=device, dim_feedforward=cfg.dim_ff,
                                                batch_first=True)
            self.t_block = CustomTransformerEncoder(block, 4, enable_nested_tensor=False)
        else:
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
        self.pos_embed = get_positional_encoding_1d(cfg.dim)
        self.prediction = nn.Sequential(
            nn.Dropout(cfg.pred_drop),
            nn.Linear(cfg.dim, 1, dtype=torch.float32)
        )
       
    def forward(self, x):
        mask = x == self.pad_idx

        t_output = self.embed(x)
        batch_size, inst_size, seq_size, dim = t_output.shape

        t_output = t_output.view(batch_size * inst_size, seq_size, dim)
        t_output = self.pos_embed(t_output)

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
        i_output = self.pos_embed(t_output)
        del t_output

        mask = mask.view(batch_size, inst_size, seq_size)
        mask = mask.all(dim=-1)

        i_output = self.t_block(i_output, src_key_padding_mask=mask)
        i_output = i_output.masked_fill(mask.unsqueeze(-1), 0)
        i_output = i_output.sum(dim = 1)
        out = self.prediction(i_output).squeeze(1)

        return out
    
    def forward_show(self, x):
        mask = x == self.pad_idx

        t_output = self.embed(x)
        batch_size, inst_size, seq_size, dim = t_output.shape

        t_output = t_output.view(batch_size * inst_size, seq_size, dim)
        t_output = self.pos_embed(t_output)

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
        i_output = self.pos_embed(t_output)
        del t_output

        mask = mask.view(batch_size, inst_size, seq_size)
        mask = mask.all(dim=-1)

        i_output, attn = self.t_block.forward_show(i_output, src_key_padding_mask=mask)
        i_output = i_output.masked_fill(mask.unsqueeze(-1), 0)
        before_sum = i_output
        i_output = i_output.sum(dim = 1)
        out = self.prediction(i_output).squeeze(1)

        return out, attn, before_sum
    