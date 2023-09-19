import torch
import torch.nn as nn
import torch.nn.functional as F

from .CustomSelfAttention import CustomSelfAttention
    
class DeepPMTransformerEncoderLayer(nn.Module):
    def __init__(self, dim, n_heads, dim_ff=2048, use_layernorm=False, layer_norm_eps=1e-05, dropout=None):
        super().__init__()

        if dropout is None:
            dropout = 0.0

        self.attn = CustomSelfAttention(dim, n_heads, dropout)
        self.proj = nn.Linear(dim, dim)

        self.pwff = nn.Sequential(
            nn.Linear(dim, dim_ff),
            nn.GELU(),
            nn.Linear(dim_ff, dim)
        )

        self.dropout = nn.Dropout(dropout)

        self.pwff = nn.Sequential(
            nn.Linear(dim, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, dim),
            nn.Dropout(dropout)
        )

        self.use_layernorm = use_layernorm
        if self.use_layernorm:
            self.norm1 = nn.LayerNorm(dim, eps=layer_norm_eps)
            self.norm2 = nn.LayerNorm(dim, eps=layer_norm_eps)

    def forward(self, src, src_key_padding_mask, weighted_attn=None):
        x = src
        
        h = self.attn(x, key_padding_mask=src_key_padding_mask, attn_mask_modifier=weighted_attn)
        h = self.dropout(h)

        if self.use_layernorm:
            h = self.norm1(x + self.proj(h))
            h = self.norm2(h + self.pwff(h))
        else:
            h = x + self.proj(h)
            h = h + self.pwff(h)

        return h
