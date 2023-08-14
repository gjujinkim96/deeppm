import torch
import torch.nn as nn
from utils import get_device
from losses import load_losses
from torch.utils.checkpoint import checkpoint

from .base_class import CheckpointModule
from .pos_encoder import get_positional_encoding_1d

def method_dummy_wrapper(func):
    def func_with_dummy(x, dummy):
        return func(x)
    return func_with_dummy

def method_dummy_wrapper2(func):
    def func_with_dummy(x, data, dummy):
        return func(x, data)
    return func_with_dummy

class PadZeroCheckpoint(CheckpointModule):
    """DeepPM model with Trasformer """
    def __init__(self, use_checkpoint=False,
                dim=512, n_heads=8, dim_ff=2048, 
                pad_idx=628, vocab_size=700, pred_drop=0.0,
                num_basic_block_layer=2,
                num_instruction_layer=2,
                num_op_layer=4, loss_type='MapeLoss', loss_fn_arg={}, checkpoint_cnt=2):
        super().__init__(use_checkpoint=use_checkpoint)

        self.use_checkpoint = use_checkpoint
        self.checkpoint_cnt = checkpoint_cnt
        self.num_basic_block_layer = num_basic_block_layer
        self.num_instruction_layer = num_instruction_layer
        self.num_op_layer = num_op_layer

        device = get_device(should_print=False)

        if self.num_basic_block_layer > 0:
            if self.use_checkpoint:
                self.basic_block_layers = nn.ModuleList()
                left = self.num_basic_block_layer
                for idx in range((self.num_basic_block_layer + self.checkpoint_cnt - 1)//self.checkpoint_cnt):
                    using_layer_cnt = min(left, self.checkpoint_cnt)
                    block = nn.TransformerEncoderLayer(dim, n_heads, device=device, dim_feedforward=dim_ff,
                                                        batch_first=True)
                    basic_block_layer = nn.TransformerEncoder(block, using_layer_cnt, enable_nested_tensor=False)
                    self.basic_block_layers.append(basic_block_layer)

                    left -= self.checkpoint_cnt
            else:
                block = nn.TransformerEncoderLayer(dim, n_heads, device=device, dim_feedforward=dim_ff,
                                                    batch_first=True)
                self.basic_block_layers = nn.ModuleList([nn.TransformerEncoder(block, self.num_basic_block_layer, enable_nested_tensor=False)])

        if self.num_instruction_layer > 0:
            if self.use_checkpoint:
                self.instruction_layers = nn.ModuleList()
                left = self.num_instruction_layer
                for idx in range((self.num_instruction_layer + self.checkpoint_cnt - 1)//self.checkpoint_cnt):
                    using_layer_cnt = min(left, self.checkpoint_cnt)

                    block = nn.TransformerEncoderLayer(dim, n_heads, device=device, dim_feedforward=dim_ff,
                                                        batch_first=True)
                    instruction_layer = nn.TransformerEncoder(block, using_layer_cnt, enable_nested_tensor=False)
                    self.instruction_layers.append(instruction_layer)

                    left -= self.checkpoint_cnt
            else:
                block = nn.TransformerEncoderLayer(dim, n_heads, device=device, dim_feedforward=dim_ff,
                                                        batch_first=True)
                self.instruction_layers = nn.ModuleList([nn.TransformerEncoder(block, self.num_instruction_layer, enable_nested_tensor=False)])

        if self.num_op_layer > 0:
            if self.use_checkpoint:
                self.op_layers = nn.ModuleList()
                left = self.num_op_layer
                for idx in range((self.num_op_layer + self.checkpoint_cnt - 1)//self.checkpoint_cnt):
                    using_layer_cnt = min(left, self.checkpoint_cnt)

                    block = nn.TransformerEncoderLayer(dim, n_heads, device=device, dim_feedforward=dim_ff,
                                                        batch_first=True)
                    op_layer = nn.TransformerEncoder(block, using_layer_cnt, enable_nested_tensor=False)
                    self.op_layers.append(op_layer)

                    left -= self.checkpoint_cnt
            else:
                block = nn.TransformerEncoderLayer(dim, n_heads, device=device, dim_feedforward=dim_ff,
                                                        batch_first=True)
                self.op_layers = nn.ModuleList([nn.TransformerEncoder(block, self.num_op_layer, enable_nested_tensor=False)])

        self.pad_idx = pad_idx
        self.embed = nn.Embedding(vocab_size, dim, padding_idx = pad_idx,
                                dtype=torch.float32, device=device) # token embedding
        self.pos_embed = get_positional_encoding_1d(dim)
        self.prediction = nn.Sequential(
            nn.Dropout(pred_drop),
            nn.Linear(dim, 1, dtype=torch.float32)
        )

        self.loss = load_losses(loss_type, loss_fn_arg)

        if self.use_checkpoint:
            self.dummy = torch.zeros(1, requires_grad=True, device=device)
    
    def checkpoint_forward(self, x):
        output = checkpoint(method_dummy_wrapper(self._basic_setup), x, self.dummy)

        if self.num_basic_block_layer > 0:
            for idx in range((self.num_basic_block_layer + self.checkpoint_cnt - 1)//self.checkpoint_cnt):
                output = checkpoint(method_dummy_wrapper2(self._basic_block_layer), output, idx, self.dummy)

        
        
        if self.num_instruction_layer > 0:
            for idx in range((self.num_instruction_layer + self.checkpoint_cnt - 1)//self.checkpoint_cnt):
                output = checkpoint(method_dummy_wrapper2(self._instruction_layer), output, idx, self.dummy)

        output, (batch_size, inst_size, seq_size), mask, op_seq_mask = output
        output = output.view(batch_size, inst_size, seq_size, -1)
        op_seq_mask = op_seq_mask.view(batch_size, inst_size)
        output = output[:,:, 0,:]
        output = output, (batch_size, inst_size, seq_size), mask, op_seq_mask

        if self.num_op_layer > 0:
            for idx in range((self.num_op_layer + self.checkpoint_cnt - 1)//self.checkpoint_cnt):
                output = checkpoint(method_dummy_wrapper2(self._op_layer), output, idx, self.dummy)

        return checkpoint(method_dummy_wrapper(self._pred), output, self.dummy)

    def _basic_setup(self, x):
        # Basic setup
        batch_size, inst_size, seq_size = x.shape
        mask = x == self.pad_idx
        op_seq_mask = mask.all(dim=-1)
        output = self.embed(x)

        # Adding pos emb
        output = output.view(batch_size * inst_size, seq_size, -1)
        output = self.pos_embed(output)

        return output, (batch_size, inst_size, seq_size), mask, op_seq_mask
    
    def _basic_block_layer(self, x, idx):
        output, (batch_size, inst_size, seq_size), mask, op_seq_mask = x
        
        if idx == 0:
            output = output.view(batch_size, inst_size * seq_size, -1)
            mask = mask.view(batch_size, inst_size * seq_size)

        output = self.basic_block_layers[idx](output, src_key_padding_mask=mask)
        return output, (batch_size, inst_size, seq_size), mask, op_seq_mask 

    def _instruction_layer(self, x, idx):
        output, (batch_size, inst_size, seq_size), mask, op_seq_mask = x
        
        if idx == 0:
            output = output.view(batch_size * inst_size, seq_size, -1)
            mask = mask.view(batch_size * inst_size, seq_size)
            op_seq_mask = op_seq_mask.view(batch_size * inst_size)
            output = output.masked_fill(op_seq_mask.unsqueeze(-1).unsqueeze(-1), 1)

        mod_mask = mask.masked_fill(op_seq_mask.unsqueeze(-1), False)
        output = self.instruction_layers[idx](output, src_key_padding_mask=mod_mask)
        
        if idx + 1 == (self.num_instruction_layer + self.checkpoint_cnt - 1)//self.checkpoint_cnt:
            output = output.masked_fill(op_seq_mask.unsqueeze(-1).unsqueeze(-1), 0)

        return output, (batch_size, inst_size, seq_size), mask, op_seq_mask 

    def _op_layer(self, x, idx):
        output, (batch_size, inst_size, seq_size), mask, op_seq_mask = x

        if idx == 0:
            output = self.pos_embed(output)

        output = self.op_layers[idx](output, src_key_padding_mask=op_seq_mask)
        return output, (batch_size, inst_size, seq_size), mask, op_seq_mask 

    def _pred(self, x):
        output, (batch_size, inst_size, seq_size), mask, op_seq_mask = x
        output = output.masked_fill(op_seq_mask.unsqueeze(-1), 0)
        output = output.sum(dim = 1)
        out = self.prediction(output).squeeze(1)
        return out
     
    def forward(self, x):
        output = self._basic_setup(x)

        if self.num_basic_block_layer > 0:
            output = self._basic_block_layer(output, 0)

        if self.num_instruction_layer > 0:
            output = self._instruction_layer(output, 0)

        output, (batch_size, inst_size, seq_size), mask, op_seq_mask = output
        output = output.view(batch_size, inst_size, seq_size, -1)
        op_seq_mask = op_seq_mask.view(batch_size, inst_size)
        output = output[:,:, 0,:]
        output = output, (batch_size, inst_size, seq_size), mask, op_seq_mask

        
        if self.num_op_layer > 0:
            output = self._op_layer(output, 0)

        return self._pred(output)


    def get_loss(self):
        return self.loss
    