import torch
import torch.nn as nn
from utils import get_device
from losses import load_losses

from .base_class import BaseModule
from .pos_encoder import get_positional_encoding_1d, get_positional_encoding_2d
from .base import Seq, Op, BasicBlock

class BaselineTest11(BaseModule):
    """DeepPM model with Trasformer """
    def __init__(self, 
                dim=512, n_heads=8, dim_ff=2048, 
                pad_idx=628, vocab_size=700, pred_drop=0.0,
                num_basic_block_layer=2,
                num_instruction_layer=2,
                num_op_layer=4, loss_type='MapeLoss', loss_fn_arg={},
                start_idx=5, end_idx=6, dsts_idx=2, srcs_idx=1,
                mem_idx=3, mem_end_idx=4, t_dropout=0.1):
        super().__init__()

        self.start_idx = start_idx
        self.end_idx = end_idx
        self.dsts_idx = dsts_idx
        self.srcs_idx = srcs_idx
        self.mem_idx = mem_idx
        self.mem_end_idx = mem_end_idx

        self.num_basic_block_layer = num_basic_block_layer
        self.num_instruction_layer = num_instruction_layer
        self.num_op_layer = num_op_layer

        device = get_device(should_print=False)

        # 1
        self.single = Seq(dim, dim_ff, n_heads, 2)

        # 2
        self.mixed = BasicBlock(dim, dim_ff, n_heads, 2)

        # 3
        self.t_single = Seq(dim, dim_ff, n_heads, 1)
        self.t_mixed = BasicBlock(dim, dim_ff, n_heads, 1)
        self.t_op = Op(dim, dim_ff, n_heads, 1)

        # 4
        self.f_bb = BasicBlock(dim, dim_ff, n_heads, 1)
        self.f_seq = Seq(dim, dim_ff, n_heads, 1)

        # 5
        self.op_only = Op(dim, dim_ff, n_heads, 2)

        # 6
        self.six_seq = Seq(dim, dim_ff, n_heads, 1)
        self.six_op = Op(dim, dim_ff, n_heads, 1)

        # 7
        self.seven_seq = Seq(dim, dim_ff, n_heads, 1)
        self.seven_bb = BasicBlock(dim, dim_ff, n_heads, 1)

        # 8
        self.eight_bb = BasicBlock(dim, dim_ff, n_heads, 1)
        self.eight_op = Op(dim, dim_ff, n_heads, 1)

        # 9
        self.nine_bb = BasicBlock(dim, dim_ff, n_heads, 1)
        self.nine_seq = Seq(dim, dim_ff, n_heads, 1)
        self.nine_op = Op(dim, dim_ff, n_heads, 1)

        self.pad_idx = pad_idx
        self.embed = nn.Embedding(vocab_size, dim, padding_idx = pad_idx,
                                dtype=torch.float32, device=device) # token embedding
    
        self.pos_embed = get_positional_encoding_1d(dim)
        self.pos2d_embed = get_positional_encoding_2d(dim)

        self.prediction = nn.Sequential(
            nn.Dropout(pred_drop),
            nn.Linear(dim, 1, dtype=torch.float32)
        )

        self.merger = nn.Sequential(
            nn.Dropout(),
            nn.Linear(9 * dim, dim)
        )

        self.loss = load_losses(loss_type, loss_fn_arg)
       
    def forward(self, x):
        # Basic setup
        batch_size, inst_size, seq_size = x.shape
        mask = x == self.pad_idx
        op_seq_mask = mask.all(dim=-1)
        output = self.embed(x)

        output = self.pos2d_embed(output)

        # Mixed
        mixed_output = self.mixed(output, mask)
        single_out = self.single(output, mask, op_seq_mask)
    
        t_out = self.t_single(output, mask, op_seq_mask)
        t_out = self.t_mixed(t_out, mask)
        t_out = t_out.sum(dim=2)
        t_out = self.pos_embed(t_out)
        t_out = self.t_op(t_out, op_seq_mask)

        # four
        f_out = self.f_bb(output, mask)
        f_out = self.f_seq(f_out, mask, op_seq_mask)


        #  5
        op_only_out = output.masked_fill(mask.unsqueeze(-1), 0)
        op_only_out = op_only_out.sum(2)
        op_only_out = self.pos_embed(op_only_out)
        op_only_out = self.op_only(op_only_out, op_seq_mask)

        #  6
        six_out = self.six_seq(output, mask, op_seq_mask)
        six_out = six_out.sum(dim=2)
        six_out = self.pos_embed(six_out)
        six_out = self.six_op(six_out, op_seq_mask)

        # 7
        seven_out = self.seven_seq(output, mask, op_seq_mask)
        seven_out = self.seven_bb(seven_out, mask)

        # 8
        eight_out = self.eight_bb(output, mask)
        eight_out = eight_out.sum(dim=2)
        eight_out = self.pos_embed(eight_out)
        eight_out = self.eight_op(eight_out, op_seq_mask)

        # 9
        nine_out = self.nine_bb(output, mask)
        nine_out = self.nine_seq(nine_out, mask, op_seq_mask)
        nine_out = nine_out.sum(dim=2)
        nine_out = self.pos_embed(nine_out)
        nine_out = self.nine_op(nine_out, op_seq_mask)

        #  Merging
        single_out = single_out.sum(dim=2)
        mixed_output = mixed_output.sum(dim=2)
        f_out = f_out.sum(dim=2)
        seven_out = seven_out.sum(dim=2)


        adding = [single_out, mixed_output, t_out, f_out, op_only_out, six_out,
                    seven_out, eight_out, nine_out]
        output = torch.stack(adding, dim=2)
        output = output.view(batch_size, inst_size, -1)
        output = self.merger(output)
        output = self.prediction(output)
        output = output.masked_fill(op_seq_mask.unsqueeze(-1), 0)
        output = output.squeeze(-1)
        output = output.sum(dim = 1)

        return output
  
    def get_loss(self):
        return self.loss
    