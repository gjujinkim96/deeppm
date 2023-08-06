import torch
import torch.nn as nn
from utils import get_device

from .pos_encoder import get_positional_encoding_1d

class OpSrcDest(nn.Module):
    """DeepPM model with Trasformer """
    def __init__(self, cfg):
        super().__init__()

        device = get_device(should_print=False)
        block = nn.TransformerEncoderLayer(cfg.dim, cfg.n_heads, device=device, dim_feedforward=cfg.dim_ff,
                                            batch_first=True)
        self.f_block = nn.TransformerEncoder(block, 2, enable_nested_tensor=False)

        block = nn.TransformerEncoderLayer(cfg.dim, cfg.n_heads, device=device, dim_feedforward=cfg.dim_ff,
                                            batch_first=True)
        self.s_block = nn.TransformerEncoder(block, 2, enable_nested_tensor=False)

        block = nn.TransformerEncoderLayer(cfg.dim, cfg.n_heads, device=device, dim_feedforward=cfg.dim_ff,
                                            batch_first=True)
        self.t_block = nn.TransformerEncoder(block, 4, enable_nested_tensor=False)


        self.pad_idx = cfg.pad_idx
        self.src_idx = cfg.src_idx
        self.dst_idx = cfg.dst_idx

        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim, padding_idx = cfg.pad_idx,
                                dtype=torch.float32, device=device) # token embedding
        self.pos_embed = get_positional_encoding_1d(cfg.dim)

        self.prediction = nn.Sequential(
            nn.Dropout(cfg.pred_drop),
            nn.Linear(cfg.dim, 1, dtype=torch.float32)
        )
       
    def forward(self, x):
        mask = x == self.pad_idx
        src = x == self.src_idx
        dst = x == self.dst_idx

        bad = src.any(dim=2).logical_not()
        bad_mask = torch.cat(
            [torch.zeros(src.size(0), src.size(1), src.size(2)-1, dtype=torch.bool, device=x.device), bad.unsqueeze(2)], dim=2)
        src = src.logical_or(bad_mask)

        bad = dst.any(dim=2).logical_not()
        bad_mask = torch.cat([torch.zeros(dst.size(0), dst.size(1), dst.size(2)-1, dtype=torch.bool, device=x.device), bad.unsqueeze(2)], dim=2)
        dst = dst.logical_or(bad_mask)

        t_output = self.embed(x)
        batch_size, inst_size, seq_size, dim = t_output.shape
        
        t_output = t_output.view(batch_size * inst_size, seq_size, dim)
        t_output = self.pos_embed(t_output)

        t_output = t_output.view(batch_size, inst_size * seq_size, dim)
        mask = mask.view(batch_size, inst_size * seq_size)

        t_output = self.f_block(t_output, src_key_padding_mask=mask)
        t_output = t_output.view(batch_size * inst_size, seq_size, dim)
        mask = mask.view(batch_size * inst_size, seq_size)

        t_output = t_output.masked_fill(mask.all(dim=-1).unsqueeze(-1).unsqueeze(-1), 1)

        mod_mask = mask.masked_fill(mask.all(dim=-1).unsqueeze(-1), False)

        t_output = self.s_block(t_output, src_key_padding_mask=mod_mask)
        

        t_output = t_output.masked_fill(mask.all(dim=-1).unsqueeze(-1).unsqueeze(-1), 0)
        t_output = t_output.view(batch_size, inst_size, seq_size, dim)

        op_output = t_output[:, :, 0, :].unsqueeze(2)
        src_output = t_output[src].view(batch_size, inst_size, dim).unsqueeze(2)
        dst_output = t_output[dst].view(batch_size, inst_size, dim).unsqueeze(2)

        t_output = torch.cat((op_output, src_output, dst_output), dim=2)
        del op_output
        del src_output
        del dst_output

        #  batch, inst, 3, dim

        t_output = t_output.view(batch_size, inst_size * 3 , dim)
        

        i_output = self.pos_embed(t_output)
        del t_output

        mask = mask.view(batch_size, inst_size, seq_size)
        mask = mask.all(dim=-1).unsqueeze(2)
        mask = mask.expand(batch_size, inst_size, 3)
        mask = mask.reshape(batch_size, inst_size * 3)

        i_output = self.t_block(i_output, src_key_padding_mask=mask)
        i_output = i_output.masked_fill(mask.unsqueeze(-1), 0)

        i_output = i_output.view(batch_size, inst_size, 3, dim)

        #  B INST DIM
        i_output = i_output[:, :, 0, :]
        
        i_output = i_output.sum(dim = 1)
        out = self.prediction(i_output).squeeze(1)

        return out
    