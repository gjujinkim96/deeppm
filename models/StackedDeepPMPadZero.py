import torch
import torch.nn as nn
from .transformer_show_attention import CustomTransformerEncoderLayer, CustomTransformerEncoder
from utils import get_device
from losses import load_losses

from .base_class import BaseModule
from .pos_encoder import get_positional_encoding_1d

class StackedDeepPMPadZero(BaseModule):
    """DeepPM model with Trasformer """
    def __init__(self, 
                dim=512, n_heads=8, dim_ff=2048, 
                pad_idx=628, vocab_size=700, pred_drop=0.0,
                num_basic_block_layer=2,
                num_instruction_layer=2,
                num_op_layer=4, loss_type='MapeLoss', loss_fn_arg={}):
        super().__init__()

        device = get_device(should_print=False)
        block = nn.TransformerEncoderLayer(dim, n_heads, device=device, dim_feedforward=dim_ff,
                                            batch_first=True)
        self.basic_block_layer = nn.TransformerEncoder(block, num_basic_block_layer, enable_nested_tensor=False)

        block = nn.TransformerEncoderLayer(dim, n_heads, device=device, dim_feedforward=dim_ff,
                                            batch_first=True)
        self.instruction_layer = nn.TransformerEncoder(block, num_instruction_layer, enable_nested_tensor=False)

        block = nn.TransformerEncoderLayer(dim, n_heads, device=device, dim_feedforward=dim_ff,
                                            batch_first=True)
        self.op_layer = nn.TransformerEncoder(block, num_op_layer, enable_nested_tensor=False)


        self.pad_idx = pad_idx
        self.embed = nn.Embedding(vocab_size, dim, padding_idx = pad_idx,
                                dtype=torch.float32, device=device) # token embedding
        self.pos_embed = get_positional_encoding_1d(dim)
        self.prediction = nn.Sequential(
            nn.Dropout(pred_drop),
            nn.Linear(dim, 1, dtype=torch.float32)
        )

        self.loss = load_losses(loss_type, loss_fn_arg)
       
    def forward(self, x):
        # Basic setup
        batch_size, inst_size, seq_size = x.shape
        mask = x == self.pad_idx
        op_seq_mask = mask.all(dim=-1)
        output = self.embed(x)

        # Adding pos emb
        output = output.view(batch_size * inst_size, seq_size, -1)
        output = self.pos_embed(output)

        # Basic block layer
        output = output.view(batch_size, inst_size * seq_size, -1)
        mask = mask.view(batch_size, inst_size * seq_size)
        output = self.basic_block_layer(output, src_key_padding_mask=mask)

        # Instruction layer
        output = output.view(batch_size * inst_size, seq_size, -1)
        mask = mask.view(batch_size * inst_size, seq_size)
        op_seq_mask = op_seq_mask.view(batch_size * inst_size)

        output = output.masked_fill(op_seq_mask.unsqueeze(-1).unsqueeze(-1), 1)
        mod_mask = mask.masked_fill(op_seq_mask.unsqueeze(-1), False)
        output = self.instruction_layer(output, src_key_padding_mask=mod_mask)
        output = output.masked_fill(op_seq_mask.unsqueeze(-1).unsqueeze(-1), 0)


        output = output[:,0,:]
        output = output.view(batch_size, inst_size, -1)
        output = self.pos_embed(output)

        op_seq_mask = op_seq_mask.view(batch_size, inst_size)

        output = self.op_layer(output, src_key_padding_mask=op_seq_mask)
        output = output.masked_fill(op_seq_mask.unsqueeze(-1), 0)
        output = output.sum(dim = 1)
        out = self.prediction(output).squeeze(1)

        return out
    
    def get_loss(self):
        return self.loss
    