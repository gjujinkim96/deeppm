import torch
import torch.nn as nn
from utils import get_device
from losses import load_losses

from .base_class import BaseModule
from .pos_encoder import get_positional_encoding_1d, get_positional_encoding_2d
from .base import Seq, Op, BasicBlock


class BertPM(BaseModule):
    def __init__(self, pretrained=None, pad_idx=0, loss_type='MapeLoss', loss_fn_arg={}, pred_drop=0.1):
        super().__init__()

        self.pt_embed = pretrained.embed
        self.pt_pos2d_embed = pretrained.pos2d_embed
        self.pt_mixed = pretrained.mixed

        dim = pretrained.embed.embedding_dim
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

        output = self.pt_embed(x)
        output = self.pt_pos2d_embed(output)

        # Mixed = B I S D
        output = self.pt_mixed(output, mask)
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
    
    def get_loss(self):
        return self.loss
    