import torch
import torch.nn as nn
from utils import get_device
from losses import load_losses

from .base_class import CheckpointModule
from .pos_encoder import get_positional_encoding_1d, get_positional_encoding_2d
from .base import Seq, Op, BasicBlock

class InstBlockOp(CheckpointModule):
    def __init__(self, use_checkpoint=False,
                dim=512, n_heads=8, dim_ff=2048, 
                pad_idx=0, vocab_size=700, pred_drop=0.0,
                num_basic_block_layer=2,
                num_instruction_layer=2,
                num_op_layer=4, loss_type='MapeLoss', loss_fn_arg={},
                t_dropout=0.1):
        super().__init__(use_checkpoint=use_checkpoint)

        self.pad_idx = pad_idx
        device = get_device(should_print=False)

        self.instr = Seq(dim, dim_ff, n_heads, num_instruction_layer, t_dropout, use_checkpoint=self.use_checkpoint)
        self.basic_block = BasicBlock(dim, dim_ff, n_heads, num_instruction_layer, t_dropout, use_checkpoint=self.use_checkpoint)
        self.op = Op(dim, dim_ff, n_heads, num_instruction_layer, t_dropout, use_checkpoint=self.use_checkpoint)

        self.embed = nn.Embedding(vocab_size, dim, padding_idx = pad_idx) # token embedding

        self.pos_embed = get_positional_encoding_1d(dim)
        self.pos_embed2 = get_positional_encoding_2d(dim)
        self.prediction = nn.Sequential(
            nn.Dropout(pred_drop),
            nn.Linear(dim, 1)
        )

        self.loss = load_losses(loss_type, loss_fn_arg)
       
    def forward(self, x):
        batch_size, inst_size, seq_size = x.shape
        mask = x == self.pad_idx
        op_seq_mask = mask.all(dim=-1)

        output = self.embed(x)

        output = output.view(batch_size * inst_size, seq_size, -1)
        output = self.pos_embed(output)
        output = output.view(batch_size, inst_size, seq_size, -1)

        output = self.instr(output, mask, op_seq_mask)

        output = self.pos_embed2(output)
        output = self.basic_block(output, mask)

        output = output.sum(dim=2)
        output = self.pos_embed(output)
        output = self.op(output, op_seq_mask)
        output = self.prediction(output).squeeze(-1)
        output = output.sum(dim=1)
        return output

    def checkpoint_forward(self, x):
        batch_size, inst_size, seq_size = x.shape
        mask = x == self.pad_idx
        op_seq_mask = mask.all(dim=-1)

        output = self.embed(x)

        output = output.view(batch_size * inst_size, seq_size, -1)
        output = self.pos_embed(output)
        output = output.view(batch_size, inst_size, seq_size, -1)

        output = self.instr.checkpoint_forward(output, mask, op_seq_mask)
        output = self.pos_embed2(output)
        output = self.basic_block.checkpoint_forward(output, mask)

        output = output.sum(dim=2)
        output = self.pos_embed(output)
        output = self.op.checkpoint_forward(output, op_seq_mask)
        output = self.prediction(output).squeeze(-1)
        output = output.sum(dim=1)
        return output
    