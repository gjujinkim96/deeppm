import torch
import torch.nn as nn

from .pos_encoder import get_positional_encoding_1d
from .deeppm_basic_blocks import DeepPMBasicBlock, DeepPMSeq, DeepPMOp
from utils import get_device

class NoNorm224(nn.Module):
    def __init__(self,
                dim=512, n_heads=8, dim_ff=2048, 
                pad_idx=628, vocab_size=700,
                num_basic_block_layer=2,
                num_instruction_layer=2,
                num_op_layer=4, 
                use_checkpoint=False,
                activation='relu',
                pred_drop=0.1, dropout=0.1):
        super().__init__()

        self.use_checkpoint = use_checkpoint
        if self.use_checkpoint:
            device = get_device(should_print=False)
            self.dummy = torch.zeros(1, requires_grad=True, device=device)
        else:
            self.dummy = None
        

        self.basic_block = DeepPMBasicBlock(dim, dim_ff, n_heads, num_basic_block_layer, 
                                dropout=dropout, use_layernorm=False, 
                                use_checkpoint=use_checkpoint, dummy=self.dummy)
        self.instruction_block = DeepPMSeq(dim, dim_ff, n_heads, num_instruction_layer,
                                dropout=dropout, use_layernorm=False, 
                                use_checkpoint=use_checkpoint, dummy=self.dummy)
        self.op_block = DeepPMOp(dim, dim_ff, n_heads, num_op_layer,
                                dropout=dropout, use_layernorm=False, 
                                use_checkpoint=use_checkpoint, dummy=self.dummy)


        self.pad_idx = pad_idx
        self.embed = nn.Embedding(vocab_size, dim, padding_idx = pad_idx)
        self.pos_embed = get_positional_encoding_1d(dim)
        self.prediction = nn.Sequential(
            nn.Dropout(pred_drop),
            nn.Linear(dim, 1)
        )
    
    def forward(self, x):
        batch_size, inst_size, seq_size = x.shape
        mask = x == self.pad_idx
        op_seq_mask = mask.all(dim=-1)

        # Embbedding
        # B I S H
        output = self.embed(x)

        # B*I S H
        output = output.view(batch_size * inst_size, seq_size, -1)
        output = self.pos_embed(output)

        # B I S H
        output = output.view(batch_size, inst_size, seq_size, -1)

        output = self.basic_block(output, mask)
        output = self.instruction_block(output, mask, op_seq_mask)

        # Reduction
        # B I H
        output = output.sum(dim=2)
        output = self.pos_embed(output)

        output = self.op_block(output, op_seq_mask)
        
        # B I H => B I 1 => B I => B
        output = self.prediction(output).squeeze(2)
        output = output.masked_fill(op_seq_mask, 0)
        output = output.sum(dim = 1)

        return output
    