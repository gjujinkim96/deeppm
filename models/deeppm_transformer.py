import torch
import torch.nn as nn
import torch.nn.functional as F
    
class DeepPMTransformerEncoderLayer(nn.Module):
    def __init__(self, dim, n_heads, dim_ff=2048, use_layernorm=False, layer_norm_eps=1e-05, dropout=None,
                use_weighted_attn=True, is_continue_padding=True):
        super().__init__()

        self.use_weighted_attn = use_weighted_attn
        self.is_continue_padding = is_continue_padding
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.proj = nn.Linear(dim, dim)

        self.pwff = nn.Sequential(
            nn.Linear(dim, dim_ff),
            nn.GELU(),
            nn.Linear(dim_ff, dim)
        )

        if dropout is not None:
            self.use_dropout = True
            self.dropout = nn.Dropout(dropout)

            self.pwff = nn.Sequential(
                nn.Linear(dim, dim_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim_ff, dim),
                nn.Dropout(dropout)
            )
            
        else:
            self.use_dropout = False
            self.pwff = nn.Sequential(
                nn.Linear(dim, dim_ff),
                nn.GELU(),
                nn.Linear(dim_ff, dim)
            )

        self.use_layernorm = use_layernorm
        if self.use_layernorm:
            self.norm1 = nn.LayerNorm(dim, eps=layer_norm_eps)
            self.norm2 = nn.LayerNorm(dim, eps=layer_norm_eps)

    def make_attn_mask(self, mask):
        '''
            code from chatgpt
            asked for cleaner version from deeppm_original_transformer code
        '''

        sizes = (~mask).sum(dim=1)
        maximum_size = mask.size(1)

        # Initialize the masking tensor with -inf values
        all_masking = []

        # Loop through each row of the mask
        for idx, s in enumerate(sizes):
            cur_mask = ~mask[idx]
            
            i, j = torch.meshgrid(
                torch.arange(s, device=mask.device), torch.arange(s, device=mask.device), indexing='ij'
            )

            if self.is_continue_padding:
                masking = F.pad((s - abs(i - j)) / s, (0, maximum_size-s, 0, maximum_size-s), value=-float('inf'))
            else:
                tmp = torch.full((maximum_size, maximum_size), -float('inf'), device=mask.device)
                tmp[cur_mask] = F.pad((s - abs(i - j)) / s, (0, maximum_size-s), value=-float('inf'))
            
            
                masking = torch.full((maximum_size, maximum_size), -float('inf'), device=mask.device)
                masking[:, cur_mask] = tmp[:, :s]
            all_masking.append(masking)

        all_masking = torch.stack(all_masking)

        return all_masking

    def forward(self, src, src_key_padding_mask):
        x = src
        if self.use_weighted_attn:
            attn_mask = self.make_attn_mask(src_key_padding_mask)
            neg_inf_mask = attn_mask == -float('inf')
            all_neg_inf = neg_inf_mask.all(dim=-1)
            attn_mask = attn_mask.masked_fill(all_neg_inf.unsqueeze(-1), 0)
            attn_mask = attn_mask.repeat(self.attn.num_heads, 1, 1)
        else:
            attn_mask = None

        h = self.attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=src_key_padding_mask,
                           need_weights=False)[0]
        if self.use_dropout:
            h = self.dropout(h)

        if self.use_layernorm:
            h = self.norm1(x + self.proj(h))
            h = self.norm2(h + self.pwff(h))
        else:
            h = x + self.proj(h)
            h = h + self.pwff(h)

        if self.use_weighted_attn:
            h = h.masked_fill(all_neg_inf.unsqueeze(-1), 0)

        return h
    