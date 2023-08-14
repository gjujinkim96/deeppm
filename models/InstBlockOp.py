import torch
import torch.nn as nn
from utils import get_device

from .pos_encoder import get_positional_encoding_1d

class InstBlockOp(nn.Module):
    def __init__(self, dim, n_heads, dim_ff, vocab_size=700, pad_idx=0, pred_drop=0.0):
        super().__init__()

        device = get_device(should_print=False)
        block = nn.TransformerEncoderLayer(dim, n_heads, device=device, dim_feedforward=dim_ff,
                                            batch_first=True)
        self.f_block = nn.TransformerEncoder(block, 2, enable_nested_tensor=False)

        block = nn.TransformerEncoderLayer(dim, n_heads, device=device, dim_feedforward=dim_ff,
                                            batch_first=True)
        self.s_block = nn.TransformerEncoder(block, 2, enable_nested_tensor=False)

        block = nn.TransformerEncoderLayer(dim, n_heads, device=device, dim_feedforward=dim_ff,
                                            batch_first=True)
        self.t_block = nn.TransformerEncoder(block, 4, enable_nested_tensor=False)


        self.pad_idx = pad_idx
        self.embed = nn.Embedding(vocab_size, dim, padding_idx = pad_idx,
                                dtype=torch.float32, device=device) # token embedding
        self.pos_embed = get_positional_encoding_1d(dim)
        self.prediction = nn.Sequential(
            nn.Dropout(pred_drop),
            nn.Linear(dim, 1, dtype=torch.float32)
        )
       
    def forward(self, x, debug=False):
        batch_size, inst_size, seq_size = x.shape
        mask = x == self.pad_idx

        # get word emb
        t_output = self.embed(x)

        # add instruction pos emb
        t_output = t_output.view(batch_size * inst_size, seq_size, -1)
        mask = mask.view(batch_size * inst_size, seq_size)
        t_output = self.pos_embed(t_output)

        # instruction level
        t_output = t_output.masked_fill(mask.all(dim=-1).unsqueeze(-1).unsqueeze(-1), 1)
        mod_mask = mask.masked_fill(mask.all(dim=-1).unsqueeze(-1), False)
        t_output = self.f_block(t_output, src_key_padding_mask=mod_mask)
        t_output = t_output.masked_fill(mask.all(dim=-1).unsqueeze(-1).unsqueeze(-1), 0)

        # block level
        t_output = t_output.view(batch_size, inst_size * seq_size, -1)
        mask = mask.view(batch_size, inst_size * seq_size)
        t_output = self.s_block(t_output, src_key_padding_mask=mask)

        # add opcode level pos emb + opcode level
        t_output = t_output.view(batch_size, inst_size, seq_size, -1)
        t_output = t_output[:, :, 0, :]
        t_output = self.pos_embed(t_output)

        mask = mask.view(batch_size, inst_size, seq_size)
        mask = mask.all(dim=-1)

        t_output = self.t_block(t_output, src_key_padding_mask=mask)

        # pred
        t_output = t_output.masked_fill(mask.unsqueeze(-1), 0)
        t_output = t_output.sum(dim = 1)
        out = self.prediction(t_output).squeeze(-1)
        return out
