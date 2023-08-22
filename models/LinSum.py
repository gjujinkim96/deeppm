import torch
import torch.nn as nn
from utils import get_device
from losses import load_losses

from .base_class import UnRollingModule
from .pos_encoder import get_positional_encoding_1d

class LinSum(UnRollingModule):
    """DeepPM model with Trasformer """
    def __init__(self, 
                dim=512, n_heads=8, dim_ff=2048, 
                pad_idx=628, vocab_size=700, pred_drop=0.0,
                num_basic_block_layer=2,
                num_instruction_layer=2,
                num_op_layer=4, loss_type='MapeLoss', loss_fn_arg={}):
        super().__init__()

        self.num_basic_block_layer = num_basic_block_layer
        self.num_instruction_layer = num_instruction_layer
        self.num_op_layer = num_op_layer

        device = get_device(should_print=False)

        if self.num_basic_block_layer > 0:
            block = nn.TransformerEncoderLayer(dim, n_heads, device=device, dim_feedforward=dim_ff,
                                                batch_first=True)
            self.basic_block_layer = nn.TransformerEncoder(block, num_basic_block_layer, enable_nested_tensor=False)

        if self.num_instruction_layer > 0:
            block = nn.TransformerEncoderLayer(dim, n_heads, device=device, dim_feedforward=dim_ff,
                                                batch_first=True)
            self.instruction_layer = nn.TransformerEncoder(block, num_instruction_layer, enable_nested_tensor=False)

        if self.num_op_layer > 0:
            block = nn.TransformerEncoderLayer(dim, n_heads, device=device, dim_feedforward=dim_ff,
                                                batch_first=True)
            self.op_layer = nn.TransformerEncoder(block, num_op_layer, enable_nested_tensor=False)


        self.pad_idx = pad_idx
        self.embed = nn.Embedding(vocab_size, dim, padding_idx = pad_idx,
                                dtype=torch.float32, device=device) # token embedding
        self.pos_embed = get_positional_encoding_1d(dim)
        self.prediction = nn.Sequential(
            nn.Dropout(pred_drop),
            nn.Linear(dim, 1, dtype=torch.float32)
        )

        self.loss = load_losses(loss_type, loss_fn_arg)
        self.unrolled_loss = load_losses('MSELoss', {})
       
    def forward(self, x, unrolled):
        # Basic setup
        batch_size, inst_size, seq_size = x.shape
        mask = x == self.pad_idx
        op_seq_mask = mask.all(dim=-1)
        output = self.embed(x)

        # Adding pos emb
        output = output.view(batch_size * inst_size, seq_size, -1)
        output = self.pos_embed(output)

        # Basic block layer
        if self.num_basic_block_layer > 0:
            output = output.view(batch_size, inst_size * seq_size, -1)
            mask = mask.view(batch_size, inst_size * seq_size)
            output = self.basic_block_layer(output, src_key_padding_mask=mask)

        # Instruction layer
        if self.num_instruction_layer > 0:
            output = output.view(batch_size * inst_size, seq_size, -1)
            mask = mask.view(batch_size * inst_size, seq_size)
            op_seq_mask = op_seq_mask.view(batch_size * inst_size)

            output = output.masked_fill(op_seq_mask.unsqueeze(-1).unsqueeze(-1), 1)
            mod_mask = mask.masked_fill(op_seq_mask.unsqueeze(-1), False)
            output = self.instruction_layer(output, src_key_padding_mask=mod_mask)
            output = output.masked_fill(op_seq_mask.unsqueeze(-1).unsqueeze(-1), 0)

        #  Selecting Op
        output = output.view(batch_size, inst_size, seq_size, -1)
        op_seq_mask = op_seq_mask.view(batch_size, inst_size)
        output = output[:,:, 0,:]


        # Op layer
        if self.num_op_layer > 0:
            output = self.pos_embed(output)
            output = self.op_layer(output, src_key_padding_mask=op_seq_mask)


        output = self.prediction(output).squeeze(2)
        output = output.masked_fill(op_seq_mask, 0)
        out = output.sum(dim = 1)

        return out, output
        # instr_cnt = torch.sum(op_seq_mask.logical_not(), dim=1).detach()
        # return out, instr_cnt, output
    
    def forward_each(self, x):
        # Basic setup
        batch_size, inst_size, seq_size = x.shape
        mask = x == self.pad_idx
        op_seq_mask = mask.all(dim=-1)
        output = self.embed(x)

        # Adding pos emb
        output = output.view(batch_size * inst_size, seq_size, -1)
        output = self.pos_embed(output)

        # Basic block layer
        if self.num_basic_block_layer > 0:
            output = output.view(batch_size, inst_size * seq_size, -1)
            mask = mask.view(batch_size, inst_size * seq_size)
            output = self.basic_block_layer(output, src_key_padding_mask=mask)

        # Instruction layer
        if self.num_instruction_layer > 0:
            output = output.view(batch_size * inst_size, seq_size, -1)
            mask = mask.view(batch_size * inst_size, seq_size)
            op_seq_mask = op_seq_mask.view(batch_size * inst_size)

            output = output.masked_fill(op_seq_mask.unsqueeze(-1).unsqueeze(-1), 1)
            mod_mask = mask.masked_fill(op_seq_mask.unsqueeze(-1), False)
            output = self.instruction_layer(output, src_key_padding_mask=mod_mask)
            output = output.masked_fill(op_seq_mask.unsqueeze(-1).unsqueeze(-1), 0)

        #  Selecting Op
        output = output.view(batch_size, inst_size, seq_size, -1)
        op_seq_mask = op_seq_mask.view(batch_size, inst_size)
        output = output[:,:, 0,:]


        # Op layer
        if self.num_op_layer > 0:
            output = self.pos_embed(output)
            output = self.op_layer(output, src_key_padding_mask=op_seq_mask)


        output = output.masked_fill(op_seq_mask.unsqueeze(-1), 0)
        output = self.prediction(output).squeeze(2)
        out = output.sum(dim = 1)

        return out, output
    
    def get_loss(self):
        return self.loss, self.unrolled_loss
    
    