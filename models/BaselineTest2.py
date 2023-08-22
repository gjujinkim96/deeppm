import torch
import torch.nn as nn
from utils import get_device
from losses import load_losses

from .base_class import BaseModule
from .pos_encoder import get_positional_encoding_1d

class BaselineTest2(BaseModule):
    """DeepPM model with Trasformer """
    def __init__(self, 
                dim=512, n_heads=8, dim_ff=2048, 
                pad_idx=628, vocab_size=700, pred_drop=0.0,
                num_basic_block_layer=2,
                num_instruction_layer=2,
                num_op_layer=4, loss_type='MapeLoss', loss_fn_arg={},
                start_idx=5, end_idx=6, dsts_idx=2, srcs_idx=1,
                mem_idx=3, mem_end_idx=4, t_dropout=0.1):
        super().__init__()

        self.start_idx = start_idx
        self.end_idx = end_idx
        self.dsts_idx = dsts_idx
        self.srcs_idx = srcs_idx
        self.mem_idx = mem_idx
        self.mem_end_idx = mem_end_idx

        self.num_basic_block_layer = num_basic_block_layer
        self.num_instruction_layer = num_instruction_layer
        self.num_op_layer = num_op_layer

        device = get_device(should_print=False)


        block = nn.TransformerEncoderLayer(dim, n_heads, device=device, dim_feedforward=dim_ff,
                                                batch_first=True, dropout=t_dropout)
        self.single = nn.TransformerEncoder(block, 8, enable_nested_tensor=False)

        # block = nn.TransformerEncoderLayer(dim, n_heads, device=device, dim_feedforward=dim_ff,
        #                                         batch_first=True, dropout=t_dropout)
        # self.two = nn.TransformerEncoder(block, 1, enable_nested_tensor=False)


        self.pad_idx = pad_idx
        self.embed = nn.Embedding(vocab_size, dim, padding_idx = pad_idx,
                                dtype=torch.float32, device=device) # token embedding
        self.type_embed = nn.Embedding(5, dim, padding_idx=0)
        self.mem_emb = nn.Embedding(3, dim, padding_idx=0)
        self.pos_embed = get_positional_encoding_1d(dim)
        self.prediction = nn.Sequential(
            nn.Dropout(pred_drop),
            nn.Linear(dim, 1, dtype=torch.float32)
        )

        self.merger = nn.Sequential(
            nn.Dropout(),
            nn.Linear(2 * dim, dim)
        )

        self.loss = load_losses(loss_type, loss_fn_arg)
       
    def get_odsp(self, x):
        # 0 = padding
        osdp = torch.zeros_like(x) 
        # 2 = dst
        osdp[torch.cumprod(~(x==self.end_idx), dim=2) == 1] = 2

        # 3 src
        osdp[torch.cumprod(~(x == self.dsts_idx), dim=2) == 1] = 3
        
        # 4 op
        osdp[torch.cumprod(~(x == self.srcs_idx), dim=2) == 1] = 4

        # 1 = start, end 
        osdp[x == self.end_idx] = 1
        osdp[x == self.start_idx] = 1
        return osdp
    
    def get_mem_mask(self, x):
        mask = (torch.cumsum((x == 3) | (x == 4), dim=2) % 2) + 1
        mask = mask.masked_fill(x == self.pad_idx, 0)
        return mask
    
    def forward(self, x):
        # Basic setup
        batch_size, inst_size, seq_size = x.shape
        mask = x == self.pad_idx
        op_seq_mask = mask.all(dim=-1)
        output = self.embed(x)

        # osdp = self.get_odsp(x)
        # osdp_emb = self.type_embed(osdp)

        # mm = self.get_mem_mask(x)
        # mm_emb = self.mem_emb(mm)

        # output = output + osdp_emb + mm_emb

        # Adding pos emb
        output = output.view(batch_size * inst_size, seq_size, -1)
        output = self.pos_embed(output)

        output = output.view(batch_size * inst_size, seq_size, -1)
        mask = mask.view(batch_size * inst_size, seq_size)
        op_seq_mask = op_seq_mask.view(batch_size * inst_size)

        output = output.masked_fill(op_seq_mask.unsqueeze(-1).unsqueeze(-1), 1)
        mod_mask = mask.masked_fill(op_seq_mask.unsqueeze(-1), False)
        single_out = self.single(output, src_key_padding_mask=mod_mask)
        # two_out = self.two(single_out, src_key_padding_mask=mod_mask)
        single_out = single_out.masked_fill(op_seq_mask.unsqueeze(-1).unsqueeze(-1), 0)

        # two_out = two_out.masked_fill(op_seq_mask.unsqueeze(-1).unsqueeze(-1), 0)

    
        single_out = single_out.view(batch_size, inst_size, seq_size, -1)
        # two_out = two_out.view(batch_size, inst_size, seq_size, -1)
        mask = mask.view(batch_size, inst_size, seq_size)

        single_out = single_out.masked_fill(mask.unsqueeze(-1), 0)
        # two_out = two_out.masked_fill(mask.unsqueeze(-1), 0)

        # output = output[:,:, 0,:]
        single_out = single_out.sum(dim=2)
        # two_out = two_out.sum(dim=2)
        op_seq_mask = op_seq_mask.view(batch_size, inst_size)

        # # Op layer
        # if self.num_op_layer > 0:
        #     output = self.pos_embed(output)
        #     output = self.op_layer(output, src_key_padding_mask=op_seq_mask)


        output = single_out
        # output = torch.stack((single_out, two_out), dim=2)
        output = output.view(batch_size, inst_size, -1)
        # output = self.merger(output)
        output = self.prediction(output)
        output = output.masked_fill(op_seq_mask.unsqueeze(-1), 0)
        output = output.squeeze(-1)
        output = output.sum(dim = 1)

        return output
  
    def get_loss(self):
        return self.loss
    