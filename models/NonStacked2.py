import torch
import torch.nn as nn
from utils import get_device
from losses import load_losses
from torch.utils.checkpoint import checkpoint

from .base_class import CheckpointModule, BaseModule
from .pos_encoder import get_positional_encoding_1d

def method_dummy_wrapper(func):
    def func_with_dummy(x, dummy):
        return func(x)
    return func_with_dummy

def method_dummy_wrapper2(func):
    def func_with_dummy(x, data, dummy):
        return func(x, data)
    return func_with_dummy

class NonStacked2(BaseModule):
    """DeepPM model with Trasformer """
    def __init__(self, 
                dim=512, n_heads=8, dim_ff=2048, op_idx=-1, srcs_idx=-1, dsts_idx=-1,
                pad_idx=628, vocab_size=700, pred_drop=0.0,
                num_basic_block_layer=2, num_op_layer=2,
                loss_type='MapeLoss', loss_fn_arg={}):
        super().__init__()

        self.num_basic_block_layer = num_basic_block_layer
        self.num_op_layer = num_op_layer
        device = get_device(should_print=False)

        block = nn.TransformerEncoderLayer(dim, n_heads, device=device, dim_feedforward=dim_ff,
                                        batch_first=True)
        self.basic_block_layers = nn.TransformerEncoder(block, num_basic_block_layer, enable_nested_tensor=False)

        block = nn.TransformerEncoderLayer(dim, n_heads, device=device, dim_feedforward=dim_ff,
                                        batch_first=True)
        self.op_layers = nn.TransformerEncoder(block, num_op_layer, enable_nested_tensor=False)

        self.pad_idx = pad_idx
        self.op_idx = op_idx
        self.srcs_idx = srcs_idx
        self.dsts_idx = dsts_idx
        self.embed = nn.Embedding(vocab_size, dim, padding_idx = pad_idx, device=device) # token embedding
        self.pos_embed = get_positional_encoding_1d(dim)
        self.prediction = nn.Sequential(
            nn.Dropout(pred_drop),
            nn.Linear(dim, 1)
        )

        self.loss = load_losses(loss_type, loss_fn_arg)
    
    def get_loss(self):
        return self.loss
    
    def _basic_setup(self, x):
        mask = x == self.pad_idx
        op_mask = x == self.op_idx
        output = self.embed(x)
        output = self.pos_embed(output)
        return output, mask, op_mask
    
    def _basic_block(self, inputs, idx):
        output, mask, op_mask = inputs
        output = self.basic_block_layers[idx](output, src_key_padding_mask=mask)
        return output, mask, op_mask
    
    def _op(self, inputs, idx):
        output, mask, op_mask = inputs
        output = self.basic_block_layers[idx](output, src_key_padding_mask=mask)
        return output, mask, op_mask
    
    def _sum_pred(self, inputs):
        output, mask, op_mask = inputs

        output = output.masked_fill(mask.unsqueeze(-1), 0)
        output = output.sum(dim=1)
        output = self.prediction(output).squeeze(-1)
        return output
    
    def get_op_mask(self, x):
        batch_size, _ = x.shape
        op_cnts = (x == self.op_idx).sum(dim=1)
        idxify_op_cnts = op_cnts - 1
        op_mask  = torch.zeros(batch_size, op_cnts.max().item(), dtype=torch.bool)
        true_vals = torch.ones_like(idxify_op_cnts, dtype=torch.bool)
        op_mask = torch.scatter(op_mask, 1, idxify_op_cnts.view(-1, 1), true_vals.view(-1, 1)).flip(1)
        op_mask = op_mask.cumsum(1).flip(1).to(torch.bool)
        return op_mask

    def forward(self, x):
        device = x.device
        mask = x == self.pad_idx
        op_idx_mask = x == self.op_idx

        batch_size, _ = x.shape
        op_cnts = op_idx_mask.sum(dim=1)
        idxify_op_cnts = op_cnts - 1
        op_mask  = torch.zeros(batch_size, op_cnts.max().item(), dtype=torch.bool, device=device)
        true_vals = torch.ones_like(idxify_op_cnts, dtype=torch.bool, device=device)
        op_mask = torch.scatter(op_mask, 1, idxify_op_cnts.view(-1, 1), true_vals.view(-1, 1)).flip(1)
        op_mask = op_mask.cumsum(1).flip(1).to(torch.bool) 
        # True if not pad
        # True True False False
 

        #  B S D
        output = self.embed(x)
        output = self.pos_embed(output)
        output = self.basic_block_layers(output, src_key_padding_mask=mask)

        # B OP(Maximum number of op in batch) D
        board = torch.ones(batch_size, op_cnts.max().item(), output.size(2)
                           , dtype=torch.float, device=device) * self.pad_idx
        output = board.masked_scatter(op_mask.unsqueeze(-1), output[op_idx_mask])


        op_mask = ~op_mask
        output = self.op_layers(output, src_key_padding_mask=op_mask)

        # print(output)
        # B OP
        output = self.prediction(output).squeeze(-1)
        output = output.masked_fill(op_mask, 0)

        #  B
        output = output.sum(dim=1)
        return output

    