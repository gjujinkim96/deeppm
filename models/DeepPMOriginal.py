import torch
import torch.nn as nn
from utils import get_device

from .custom_transformer import PositionalEncoding, Block, Embeddings

class DeepPMOriginal(nn.Module):
    """DeepPM model with Trasformer """
    def __init__(self, cfg):
        super().__init__()

        self.pad_idx = cfg.pad_idx
        self.embed = Embeddings(cfg)

        self.pre_blocks = nn.ModuleList([Block(cfg) for _ in range(2)])#cfg.n_layers)])
        self.token_blocks = nn.ModuleList([Block(cfg) for _ in range(2)])#cfg.n_layers)])

        self.pos_embed = PositionalEncoding(cfg.dim, 400)

        self.instruction_blocks = nn.ModuleList([Block(cfg) for _ in range(4)])#cfg.n_layers)])
        self.prediction = nn.Linear(cfg.dim,1)
       
    def forward(self, x):
        # B I S
        batch_size, inst_size, seq_size = x.shape

        padding_mask = (x == self.pad_idx)

        #  B I S D
        t_output = self.embed(x)

        #  B*I S D
        t_output = t_output.view(batch_size * inst_size, seq_size, -1)
        t_output = self.pos_embed(t_output)

        t_output = t_output.view(batch_size, inst_size * seq_size, -1)

        for t_block in self.pre_blocks:
            t_output = t_block(t_output)

        t_output = t_output.view(batch_size * inst_size, seq_size, -1)

        for t_block in self.token_blocks:
            t_output = t_block(t_output)

        t_output = t_output[:,0,:]
        t_output = t_output.view(batch_size, inst_size, -1)

        i_output = self.pos_embed(t_output)
        del t_output

        for i_block in self.instruction_blocks:
            i_output = i_block(i_output)

        #  B I
        i_output = i_output.sum(dim=1)
        out = self.prediction(i_output).squeeze(1)

        return out
