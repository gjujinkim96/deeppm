import torch
import torch.nn as nn
from utils import get_device

from .custom_transformer import PositionalEncoding

class PalmDeepPM(nn.Module):
    """DeepPM model with Trasformer """
    def __init__(self, cfg):
        super().__init__()


        self.larger = nn.Sequential(
            nn.Dropout(),
            nn.Linear(128, cfg.dim, dtype=torch.float32)
        )

        device = get_device(should_print=False)
        block = nn.TransformerEncoderLayer(cfg.dim, cfg.n_heads, device=device,
                                            batch_first=True)
        self.blocks = nn.TransformerEncoder(block, cfg.n_layers)

        self.pos_embed = PositionalEncoding(cfg.dim, 500).to(device)
        self.prediction = nn.Sequential(
            nn.Dropout(),
            nn.Linear(cfg.dim, 1, dtype=torch.float32)
        )
       
    def forward(self, x, mask):
        t_output = self.larger(x)

        t_output = self.blocks(t_output, src_key_padding_mask=mask)

        t_output = t_output[:, 0, :]
        out = self.prediction(t_output).squeeze(1)

        return out
            