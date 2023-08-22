import torch
import torch.nn as nn
from utils import get_device
from losses import load_losses

from .base_class import CheckpointModule
from .pos_encoder import get_positional_encoding_1d, get_positional_encoding_2d
from torch.utils.checkpoint import checkpoint

def method_dummy_wrapper(func):
    def func_with_dummy(x, dummy):
        return func(x)
    return func_with_dummy

class StackedDeepPMPadZeroTest(CheckpointModule):
    """DeepPM model with Trasformer """
    def __init__(self, use_checkpoint=False,
                dim=512, n_heads=8, dim_ff=2048, 
                pad_idx=628, vocab_size=700, pred_drop=0.0,
                num_basic_block_layer=2,
                num_instruction_layer=2,
                num_op_layer=4, loss_type='MapeLoss', loss_fn_arg={}):
        super().__init__(use_checkpoint=use_checkpoint)

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
        self.pos2d_embed = get_positional_encoding_2d(dim)
        self.prediction = nn.Sequential(
            nn.Dropout(pred_drop),
            nn.Linear(dim, 1, dtype=torch.float32)
        )

        self.merger = nn.Sequential(
            nn.Dropout(),
            nn.Linear(3 * dim, dim)
        )

        self.loss = load_losses(loss_type, loss_fn_arg)

        if self.use_checkpoint:
            self.dummy = torch.zeros(1, requires_grad=True, device=device)
    
    def _setup_layer(self, x):
        batch_size, inst_size, seq_size = x.shape
        mask = x == self.pad_idx
        op_seq_mask = mask.all(dim=-1)
        output = self.embed(x)

        # Adding pos emb
        output = output.view(batch_size, inst_size, seq_size, -1)
        output = self.pos2d_embed(output)
        return output, mask, op_seq_mask, (batch_size, inst_size, seq_size)

    def _basic_block_layer(self, raws):
        output, mask, op_seq_mask, (batch_size, inst_size, seq_size) = raws
        output = output.view(batch_size, inst_size * seq_size, -1)
        mask = mask.view(batch_size, inst_size * seq_size)
        output = self.basic_block_layer(output, src_key_padding_mask=mask)
        output = output.masked_fill(mask.unsqueeze(-1), 0)
        output = output.view(batch_size, inst_size, seq_size, -1)
        bb_output = output.sum(dim=2)
        return output, bb_output, mask, op_seq_mask, (batch_size, inst_size, seq_size)

    def _instruction_layer(self, raws):
        output, mask, op_seq_mask, (batch_size, inst_size, seq_size) = raws
        output = output.view(batch_size * inst_size, seq_size, -1)
        mask = mask.view(batch_size * inst_size, seq_size)
        op_seq_mask = op_seq_mask.view(batch_size * inst_size)

        output = output.masked_fill(op_seq_mask.unsqueeze(-1).unsqueeze(-1), 1)
        mod_mask = mask.masked_fill(op_seq_mask.unsqueeze(-1), False)
        output = self.instruction_layer(output, src_key_padding_mask=mod_mask)
        output = output.masked_fill(op_seq_mask.unsqueeze(-1).unsqueeze(-1), 0)

        output = output.view(batch_size, inst_size, seq_size, -1)
        mask = mask.view(batch_size, inst_size, seq_size)
        output = output.masked_fill(mask.unsqueeze(-1), 0)
        output = output.sum(dim=2)
        il_output = torch.clone(output)
        return output, il_output, mask, op_seq_mask, (batch_size, inst_size, seq_size)
    
    def _op_layer(self, raws):
        output, mask, op_seq_mask, (batch_size, inst_size, seq_size) = raws
        op_seq_mask = op_seq_mask.view(batch_size, inst_size)
        output = self.pos_embed(output)
        output = self.op_layer(output, src_key_padding_mask=op_seq_mask)

        return output, mask, op_seq_mask, (batch_size, inst_size, seq_size)
    
    def _predict_layer(self, raws):
        bb_output, il_output, output, mask, op_seq_mask, (batch_size, inst_size, seq_size) = raws
        output = torch.stack((bb_output, il_output, output), dim=2)
        output = output.view(batch_size, inst_size, -1)
        output = self.merger(output)
        output = self.prediction(output).squeeze(-1) # B I

        op_seq_mask = op_seq_mask.view(batch_size, inst_size)
        output = output.masked_fill(op_seq_mask, 0)
        output = output.sum(dim=1)
        return output
    
    def _half_part(self, x):
        raw_outputs = self._setup_layer(x)
        
        # Basic block layer
        raw_outputs = self._basic_block_layer(raw_outputs)

        output, bb_output, mask, op_seq_mask, (batch_size, inst_size, seq_size) = raw_outputs
        raw_outputs = output, mask, op_seq_mask, (batch_size, inst_size, seq_size)
        
        # Instruction layer
        raw_outputs = self._instruction_layer(raw_outputs)
        output, il_output, mask, op_seq_mask, (batch_size, inst_size, seq_size) = raw_outputs
        return bb_output, il_output, output, mask, op_seq_mask, (batch_size, inst_size, seq_size)
    
    def checkpoint_forward(self, x):
        raw_outputs = checkpoint(method_dummy_wrapper(self._half_part), x, self.dummy)
        return self._predict_layer(raw_outputs)
    
    def forward(self, x):
        raw_outputs = self._half_part(x)
        return self._predict_layer(raw_outputs)

    def get_loss(self):
        return self.loss
