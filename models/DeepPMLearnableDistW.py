import torch
import torch.nn as nn

from .pos_encoder import get_positional_encoding_1d
from .deeppm_transformer import DeepPMTransformerEncoderLayer

from torch.utils.checkpoint import checkpoint
from utils import get_device
from .checkpoint_utils import method_dummy_wrapper

class DeepPMLearnableDistW(nn.Module):
    """DeepPM model with Trasformer """
    def __init__(self, dim=512, n_heads=8, dim_ff=2048, 
                pad_idx=0, vocab_size=700,
                num_basic_block_layer=2,
                num_instruction_layer=2,
                num_op_layer=4, use_checkpoint=False, 
                dist_weight_len_limit=40,
                dist_emb_dim=32):
        super().__init__()

        self.pos_embed = get_positional_encoding_1d(dim)

        self.pad_idx = pad_idx
        self.embed = nn.Embedding(vocab_size, dim, self.pad_idx)

        self.dist_weight_emb = nn.Embedding(dist_weight_len_limit * 2 + 2, dist_emb_dim, dist_weight_len_limit * 2 + 1)
        self.dist_layer = nn.Sequential(
            nn.Linear(dist_emb_dim+1, 4 * dist_emb_dim),
            nn.ReLU(),
            nn.Linear(4 * dist_emb_dim, 1),
        )

        self.num_basic_block_layer = num_basic_block_layer
        self.num_instruction_layer = num_instruction_layer
        self.num_op_layer = num_op_layer

        self.basic_block = nn.ModuleList(
            [
                DeepPMTransformerEncoderLayer(dim, n_heads, dim_ff, use_weighted_attn=False) 
                    for _ in range(self.num_basic_block_layer)
            ]
        )

        self.instruction_block = nn.ModuleList(
            [
                DeepPMTransformerEncoderLayer(dim, n_heads, dim_ff, use_weighted_attn=False) 
                    for _ in range(self.num_instruction_layer)
            ]
        )

        self.op_block = nn.ModuleList(
            [
                DeepPMTransformerEncoderLayer(dim, n_heads, dim_ff, use_weighted_attn=True) 
                    for _ in range(self.num_op_layer)
            ]
        )

        self.prediction = nn.Linear(dim, 1)

        self.use_checkpoint = use_checkpoint
        if self.use_checkpoint:
            device = get_device(should_print=False)
            self.dummy = torch.zeros(1, requires_grad=True, device=device)



    def _basic_block(self, x, mask):
        batch_size, inst_size, seq_size, _ = x.shape
        x = x.view(batch_size, inst_size * seq_size, -1)
        mask = mask.view(batch_size, inst_size * seq_size)

        for block in self.basic_block:
            if self.use_checkpoint:
                x = checkpoint(method_dummy_wrapper(block), self.dummy, x, mask)
            else:
                x = block(x, mask)
        
        x = x.masked_fill(mask.unsqueeze(-1), 0)
        x = x.view(batch_size, inst_size, seq_size, -1)
        return x
    
    def _instruction_block(self, x, mask, op_seq_mask):
        batch_size, inst_size, seq_size, _ = x.shape

        x = x.view(batch_size * inst_size, seq_size, -1)
        mask = mask.view(batch_size * inst_size, seq_size)
        op_seq_mask = op_seq_mask.view(batch_size * inst_size)

        x = x.masked_fill(op_seq_mask.unsqueeze(-1).unsqueeze(-1), 1)
        mod_mask = mask.masked_fill(op_seq_mask.unsqueeze(-1), False)

        for block in self.instruction_block:
            if self.use_checkpoint:
                x = checkpoint(method_dummy_wrapper(block), self.dummy, x, mod_mask)
            else:
                x = block(x, mod_mask)
                
        x = x.masked_fill(mask.unsqueeze(-1), 0)
        x = x.view(batch_size, inst_size, seq_size, -1)
        return x
    
    def _op_block(self, x, op_seq_mask, dist_mask, sizes):
        # dist_mask: B I I
        # sizes: B

        batch_size, inst_size, _ = dist_mask.shape

        # B I I D
        dist_masking = self.dist_weight_emb(dist_mask)

        # B 1 1 1
        sizes = sizes.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, inst_size, inst_size, -1)
        
        # B I I D+1
        dist_feat = torch.cat((dist_masking, sizes), dim=-1)

        # B I I
        dist_weight = self.dist_layer(dist_feat).squeeze(-1)
        dist_weight = dist_weight.masked_fill(op_seq_mask.unsqueeze(1) | op_seq_mask.unsqueeze(2), -float('inf'))

        for block in self.op_block:
            if self.use_checkpoint:
                x = checkpoint(method_dummy_wrapper(block), self.dummy, x, op_seq_mask, dist_weight)
            else:
                x = block(x, op_seq_mask, weighted_attn=dist_weight)
                
        x = x.masked_fill(op_seq_mask.unsqueeze(-1), 0)
        return x
       
    def forward(self, x):
        dist_mask = x['dist_mask']
        sizes = x['sizes']
        x = x['x']

        # B I S
        batch_size, inst_size, seq_size = x.shape
        mask = x == self.pad_idx
        op_seq_mask = mask.all(dim=-1)

        #  B I S D
        output = self.embed(x)

        #  B*I S D
        output = output.view(batch_size * inst_size, seq_size, -1)
        output = self.pos_embed(output)

        #  B I S D
        output = output.view(batch_size, inst_size, seq_size, -1)
        output = self._basic_block(output, mask)
        output = self._instruction_block(output, mask, op_seq_mask)

        # reduce
        # B I H
        output = output[:, :, 0]
        output = self.pos_embed(output)

        output = self._op_block(output, op_seq_mask, dist_mask, sizes)

        #  B I
        output = output.sum(dim=1)
        output = self.prediction(output).squeeze(1)
        return output
