import torch
import torch.nn as nn
from utils import get_device
from torch.utils.checkpoint import checkpoint

from .checkpoint_utils import method_dummy_wrapper
from .deeppm_transformer import DeepPMTransformerEncoderLayer

class CustomSeq(nn.Module):
    def __init__(self, dim, dim_ff, n_heads, n_layers, use_layernorm=True, dropout=0.1, use_weighted_attn=False, is_continue_padding=True, use_checkpoint=False):
        super().__init__()

        device = get_device(should_print=False)
        self.tr = nn.ModuleList(
            [
                DeepPMTransformerEncoderLayer(dim, n_heads, dim_ff, use_layernorm=use_layernorm,
                                dropout=dropout, use_weighted_attn=use_weighted_attn, is_continue_padding=is_continue_padding)
                    for _ in range(n_layers)
            ]
        )

        self.use_checkpoint = use_checkpoint
        if self.use_checkpoint:
            self.dummy = torch.zeros(1, requires_grad=True, device=device)

    def forward(self, x, mask, op_seq_mask):
        if self.use_checkpoint:
            return self._checkpoint_forward(x, mask, op_seq_mask)
        else:
            return self._forward(x, mask, op_seq_mask)
        
    def _forward(self, x, mask, op_seq_mask):
        """
        x: [batch_size, inst_size, seq_size, dim]
        mask: [batch_size, inst_size, seq_size]
        op_seq_mask: [batch_size, inst_size]
        """
        
        batch_size, inst_size, seq_size, _ = x.shape

        x = x.view(batch_size * inst_size, seq_size, -1)
        mask = mask.view(batch_size * inst_size, seq_size)
        op_seq_mask = op_seq_mask.view(batch_size * inst_size)

        x = x.masked_fill(op_seq_mask.unsqueeze(-1).unsqueeze(-1), 1)
        mod_mask = mask.masked_fill(op_seq_mask.unsqueeze(-1), False)

        for block in self.tr:
            x = self.tr(x, src_key_padding_mask=mod_mask)

        x = x.masked_fill(mask.unsqueeze(-1), 0)
        x = x.view(batch_size, inst_size, seq_size, -1)
        return x
    
    def _checkpoint_forward(self, x, mask, op_seq_mask):
        """
        x: [batch_size, inst_size, seq_size, dim]
        mask: [batch_size, inst_size, seq_size]
        op_seq_mask: [batch_size, inst_size]
        """
        
        batch_size, inst_size, seq_size, _ = x.shape

        x = x.view(batch_size * inst_size, seq_size, -1)
        mask = mask.view(batch_size * inst_size, seq_size)
        op_seq_mask = op_seq_mask.view(batch_size * inst_size)

        x = x.masked_fill(op_seq_mask.unsqueeze(-1).unsqueeze(-1), 1)
        mod_mask = mask.masked_fill(op_seq_mask.unsqueeze(-1), False)

        for block in self.tr:
            x = checkpoint(method_dummy_wrapper(block), self.dummy, x, mod_mask)

        x = x.masked_fill(mask.unsqueeze(-1), 0)
        x = x.view(batch_size, inst_size, seq_size, -1)
        return x
    
class CustomBasicBlock(nn.Module):
    def __init__(self, dim, dim_ff, n_heads, n_layers, use_layernorm=True, dropout=0.1, use_weighted_attn=False, is_continue_padding=True, use_checkpoint=False):
        super().__init__()

        device = get_device(should_print=False)        
        self.tr = nn.ModuleList(
            [
                DeepPMTransformerEncoderLayer(dim, n_heads, dim_ff, use_layernorm=use_layernorm,
                                dropout=dropout, use_weighted_attn=use_weighted_attn, is_continue_padding=is_continue_padding)
                    for _ in range(n_layers)
            ]
        )

        self.use_checkpoint = use_checkpoint
        if self.use_checkpoint:
            self.dummy = torch.zeros(1, requires_grad=True, device=device)

    def forward(self, x, mask):
        if self.use_checkpoint:
            return self._checkpoint_forward(x, mask)
        else:
            return self._forward(x, mask)
        
    def _forward(self, x, mask):
        """
        x: [batch_size, inst_size, seq_size, dim]
        mask: [batch_size, inst_size, seq_size]
        """
        batch_size, inst_size, seq_size, _ = x.shape

        x = x.view(batch_size, inst_size * seq_size, -1)
        mask = mask.view(batch_size, inst_size * seq_size)

        for block in self.tr:
            x = self.tr(x, src_key_padding_mask=mask)

        x = x.masked_fill(mask.unsqueeze(-1), 0)
        x = x.view(batch_size, inst_size, seq_size, -1)
        return x
    
    def _checkpoint_forward(self, x, mask):
        """
        x: [batch_size, inst_size, seq_size, dim]
        mask: [batch_size, inst_size, seq_size]
        """
        batch_size, inst_size, seq_size, _ = x.shape

        x = x.view(batch_size, inst_size * seq_size, -1)
        mask = mask.view(batch_size, inst_size * seq_size)

        for block in self.tr:
            x = checkpoint(method_dummy_wrapper(block), self.dummy, x, mask)
        x = x.masked_fill(mask.unsqueeze(-1), 0)
        x = x.view(batch_size, inst_size, seq_size, -1)
        return x
    
class CustomOp(nn.Module):
    def __init__(self, dim, dim_ff, n_heads, n_layers, use_layernorm=True, dropout=0.1, use_weighted_attn=False, is_continue_padding=True, use_checkpoint=False):
        super().__init__()

        device = get_device(should_print=False)
        self.tr = nn.ModuleList(
            [
                DeepPMTransformerEncoderLayer(dim, n_heads, dim_ff, use_layernorm=use_layernorm,
                                dropout=dropout, use_weighted_attn=use_weighted_attn, is_continue_padding=is_continue_padding)
                    for _ in range(n_layers)
            ]
        )

        self.use_checkpoint = use_checkpoint
        if self.use_checkpoint:
            self.dummy = torch.zeros(1, requires_grad=True, device=device)

    def forward(self, x, op_seq_mask):
        if self.use_checkpoint:
            return self._checkpoint_forward(x, op_seq_mask)
        else:
            return self._forward(x, op_seq_mask)
        
    def _forward(self, x, op_seq_mask):
        """
        x: [batch_size, inst_size, seq_size, dim]
        op_seq_mask: [batch_size, inst_size]
        """
        for block in self.tr:
            x = self.tr(x, src_key_padding_mask=op_seq_mask)
        x = x.masked_fill(op_seq_mask.unsqueeze(-1), 0)
        return x
    
    def _checkpoint_forward(self, x, op_seq_mask):
        """
        x: [batch_size, inst_size, seq_size, dim]
        op_seq_mask: [batch_size, inst_size]
        """
        for block in self.tr:
            x = checkpoint(method_dummy_wrapper(block), self.dummy, x, op_seq_mask)

        x = x.masked_fill(op_seq_mask.unsqueeze(-1), 0)
        return x
    