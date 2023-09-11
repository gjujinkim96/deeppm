import torch
import torch.nn as nn
import torch.nn.functional as F
    
class DeepPMTransformerEncoderLayer(nn.Module):
    def __init__(self, dim, n_heads, dim_ff=2048):
        super().__init__()

        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.proj = nn.Linear(dim, dim)
        self.pwff = nn.Sequential(
            nn.Linear(dim, dim_ff),
            nn.GELU(),
            nn.Linear(dim_ff, dim)
        )

    def make_attn_mask(self, mask):
        '''
            code from chatgpt
            asked for cleaner version from deeppm_original_transformer code
        '''

        sizes = (~mask).sum(dim=1)
        # Create a list to store the masking matrices
        masking_matrices = []

        # Loop through each size
        for s in sizes:
            masking = torch.full((max(sizes), max(sizes)), -float('inf'), device=mask.device)
            
            # Calculate the values based on your condition
            i, j = torch.meshgrid(
                torch.arange(s, device=mask.device), torch.arange(s, device=mask.device), indexing='ij'
            )
            masking[:s, :s] = torch.where(i < j, (s + i - j) / s, (s - i + j) / s)
            
            # Append the masking matrix to the list
            masking_matrices.append(masking)

        # Stack the list of masking matrices to create a tensor
        masking_tensor = torch.stack(masking_matrices, dim=0)
        return masking_tensor

    def forward(self, src, src_key_padding_mask, use_weighted_attn):
        x = src
        if use_weighted_attn:
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

        h = x + self.proj(h)
        h = h + self.pwff(h)

        if use_weighted_attn:
            h = h.masked_fill(all_neg_inf.unsqueeze(-1), 0)

        return h
    