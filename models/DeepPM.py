import torch
import torch.nn as nn

from .pos_encoder import get_positional_encoding_1d
from .deeppm_basic_blocks import DeepPMBasicBlock, DeepPMSeq, DeepPMOp
from utils import get_device

class DeepPM(nn.Module):
    """DeepPM model with Trasformer """
    def __init__(self, dim=512, n_heads=8, dim_ff=2048, 
                pad_idx=0, vocab_size=700,
                num_basic_block_layer=2,
                num_instruction_layer=2,
                num_op_layer=4, use_checkpoint=False, use_layernorm=False,
                use_bb_attn=True, use_seq_attn=True, use_op_attn=True):
        super().__init__()

        self.use_checkpoint = use_checkpoint
        if self.use_checkpoint:
            device = get_device(should_print=False)
            self.dummy = torch.zeros(1, requires_grad=True, device=device)
        else:
            self.dummy = None

        self.pos_embed = get_positional_encoding_1d(dim)


        self.pad_idx = pad_idx
        self.embed = nn.Embedding(vocab_size, dim, self.pad_idx)


        self.basic_block = DeepPMBasicBlock(dim, dim_ff, n_heads, num_basic_block_layer, use_layernorm=use_layernorm,
                            use_checkpoint=use_checkpoint, dummy=self.dummy)
        self.instruction_block = DeepPMSeq(dim, dim_ff, n_heads, num_instruction_layer, use_layernorm=use_layernorm,
                            use_checkpoint=use_checkpoint, dummy=self.dummy)
        self.op_block = DeepPMOp(dim, dim_ff, n_heads, num_op_layer, use_layernorm=use_layernorm,
                            use_checkpoint=use_checkpoint, dummy=self.dummy)

        self.prediction = nn.Linear(dim, 1)

        self.use_bb_attn = use_bb_attn
        self.use_seq_attn = use_seq_attn
        self.use_op_attn = use_op_attn
       
    def forward(self, x):
        bb_attn_mod = x['bb_attn_mod'] if self.use_bb_attn else None
        seq_attn_mod = x['seq_attn_mod'] if self.use_seq_attn else None
        op_attn_mod = x['op_attn_mod'] if self.use_op_attn else None
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

        output = self.basic_block(output, mask, bb_attn_mod)
        output = self.instruction_block(output, mask, op_seq_mask, seq_attn_mod)

        # reduce
        # B I H
        output = output[:, :, 0]
        output = self.pos_embed(output)

        output = self.op_block(output, op_seq_mask, op_attn_mod)

        #  B I
        output = output.sum(dim=1)
        output = self.prediction(output).squeeze(1)
        return output
