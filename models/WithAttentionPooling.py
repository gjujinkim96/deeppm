import torch
import torch.nn as nn
from utils import get_device

from .pos_encoder import get_positional_encoding_1d

# pytorch Transformer encoder 참고
# https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#TransformerEncoderLayer
class AttentionPooling(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn = nn.MultiheadAttention(cfg.dim, cfg.n_heads, 0.1, batch_first=True)

        self.linear1 = nn.Linear(cfg.dim, cfg.dim_ff)
        self.linear2 = nn.Linear(cfg.dim_ff, cfg.dim)
        self.activation = nn.ReLU()

        self.dropout = nn.Dropout(0.1)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)

        self.norm1 = nn.LayerNorm(cfg.dim)
        self.norm2 = nn.LayerNorm(cfg.dim)

    def forward(self, q, kv, key_padding_mask):
        x = self.norm1(q + self._sa_block(q, kv, key_padding_mask))
        x = self.norm2(x + self._ff_block(x))
        return x

    def _sa_block(self, q, kv, key_padding_mask):
        x = self.attn(q, kv, kv, key_padding_mask=key_padding_mask)[0]
        return self.dropout1(x)

    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class WithAttentionPooling(nn.Module):
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


        self.attn_pooling = AttentionPooling(cfg)

        self.pad_idx = cfg.pad_idx
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim, padding_idx = cfg.pad_idx,
                                dtype=torch.float32, device=device) # token embedding
        self.pos_embed = get_positional_encoding_1d(cfg.dim)
        self.prediction = nn.Sequential(
            nn.Dropout(cfg.pred_drop),
            nn.Linear(cfg.dim, 1, dtype=torch.float32)
        )
       
    def forward(self, x, debug=False):
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

        query = t_output[:,0,:]
        query = query.view(batch_size, inst_size, dim)
        kv = t_output.view(batch_size, inst_size * seq_size, dim)
        mask = mask.view(batch_size, inst_size * seq_size)

        t_output = self.attn_pooling(query, kv, mask)

        del query
        del kv

        i_output = self.pos_embed(t_output)
        del t_output

        mask = mask.view(batch_size, inst_size, seq_size)
        mask = mask.all(dim=-1)

        i_output = self.t_block(i_output, src_key_padding_mask=mask)
        i_output = i_output.sum(dim = 1)
        out = self.prediction(i_output).squeeze(1)

        return out
    