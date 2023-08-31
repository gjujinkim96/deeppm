import torch
import torch.nn as nn
from utils import get_device
from losses import load_losses

from torch.utils.checkpoint import checkpoint
from .base_class import BaseModule, BertModule
from .pos_encoder import get_positional_encoding_1d, get_positional_encoding_2d
from .base import Seq, Op, BasicBlock

class DeepBert(BertModule):
    """DeepPM model with Trasformer """
    def __init__(self,
                dim=512, n_heads=8, dim_ff=2048, 
                pad_idx=628, vocab_size=700, pred_drop=0.1,
                num_basic_block_layer=2,
                loss_type='CrossEntropyLoss', loss_fn_arg={}, t_activation='relu', t_dropout=0.1):
        super().__init__()

        loss_fn_arg['ignore_index'] =  pad_idx
        self.vocab_size = vocab_size

        self.num_basic_block_layer = num_basic_block_layer


        device = get_device(should_print=False)

        self.mixed = BasicBlock(dim, dim_ff, n_heads, self.num_basic_block_layer, t_dropout=t_dropout, activation='gelu')

        self.pad_idx = pad_idx
        self.embed = nn.Embedding(vocab_size, dim, padding_idx = pad_idx,
                                dtype=torch.float32, device=device) # token embedding
    
        self.pos_embed = get_positional_encoding_1d(dim)
        self.pos2d_embed = get_positional_encoding_2d(dim)

        self.prediction = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(pred_drop),
            nn.Linear(dim, self.vocab_size)
        )

        self.loss = load_losses(loss_type, loss_fn_arg)
     
    def forward(self, x):
        # Basic setup
        batch_size, inst_size, seq_size = x.shape
        mask = x == self.pad_idx
        output = self.embed(x)
        output = self.pos2d_embed(output)

        # Mixed
        output = self.mixed(output, mask)

        return self.prediction(output)
  
    def get_loss(self):
        return self.loss
    