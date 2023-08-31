import torch
import torch.nn as nn
from utils import get_device
from losses import load_losses

from .base_class import BaseModule
from .pos_encoder import get_positional_encoding_1d, get_positional_encoding_2d

class JustLinear(BaseModule):
    """DeepPM model with Trasformer """
    def __init__(self, 
                dim=512, n_heads=8,
                pad_idx=628, vocab_size=700, pred_drop=0.1,
                loss_type='MapeLoss', loss_fn_arg={}):
        super().__init__()


        device = get_device(should_print=False)

        self.embed = nn.Embedding(vocab_size, dim, padding_idx = pad_idx,
                                dtype=torch.float32, device=device) # token embedding
    
        self.pos_embed = get_positional_encoding_1d(dim)
        self.pos2d_embed = get_positional_encoding_2d(dim)

        self.pad_idx = pad_idx

        self.prediction = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(pred_drop),
            nn.Linear(dim, 1)
        )

        self.loss = load_losses(loss_type, loss_fn_arg)
       
    def forward(self, x):
        mask = x == self.pad_idx

        output = self.embed(x)
        output = self.pos2d_embed(output)
        output = output.masked_fill(mask.unsqueeze(-1), 0)

        # B I D
        output = output.sum(dim=2)
        op_seq_mask = mask.all(dim=-1)

        # B I
        output = self.prediction(output).squeeze(-1)
        output = output.masked_fill(op_seq_mask, 0)

        # B
        output = output.sum(dim=1)
        
        return output
    