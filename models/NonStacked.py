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

class NonStacked(CheckpointModule):
    """DeepPM model with Trasformer """
    def __init__(self, 
                dim=512, n_heads=8, dim_ff=2048, 
                pad_idx=0, vocab_size=700, pred_drop=0.0, use_checkpoint=False, checkpoint_cnt=1,
                num_basic_block_layer=2, loss_type='MapeLoss', loss_fn_arg={}):
        super().__init__()

        self.num_basic_block_layer = num_basic_block_layer
        self.use_checkpoint = use_checkpoint
        self.checkpoint_cnt = checkpoint_cnt
        device = get_device(should_print=False)

        

        self.basic_block_layers = nn.ModuleList()
        if self.use_checkpoint:
            self.dummy = torch.zeros(1, requires_grad=True, device=device)
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
            basic_block_layer = nn.TransformerEncoder(block, num_basic_block_layer, enable_nested_tensor=False)
            self.basic_block_layers.append(basic_block_layer)

        self.pad_idx = pad_idx
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
        output = self.embed(x)
        output = self.pos_embed(output)
        return output, mask
    
    def _basic_block(self, inputs, idx):
        output, mask = inputs
        output = self.basic_block_layers[idx](output, src_key_padding_mask=mask)
        return output, mask
    
    def _sum_pred(self, inputs):
        output, mask = inputs

        output = output.masked_fill(mask.unsqueeze(-1), 0)
        output = output.sum(dim=1)
        output = self.prediction(output).squeeze(-1)
        return output
    
    def checkpoint_forward(self, x):
        output, mask = checkpoint(method_dummy_wrapper(self._basic_setup), x, self.dummy)

        for idx in range((self.num_basic_block_layer + self.checkpoint_cnt - 1)//self.checkpoint_cnt):
            output, mask = checkpoint(method_dummy_wrapper2(self._basic_block), (output, mask), idx, self.dummy)

        return checkpoint(method_dummy_wrapper(self._sum_pred), (output, mask), self.dummy)

    def forward(self, x):
        # B S D
        output, mask  = self._basic_setup(x)
        output, mask = self._basic_block((output, mask), 0)
        return self._sum_pred((output, mask))

    