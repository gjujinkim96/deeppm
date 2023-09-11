import torch.nn as nn

from .pos_encoder import get_positional_encoding_1d
from .deeppm_transformer import DeepPMTransformerEncoderLayer

class DeepPM(nn.Module):
    """DeepPM model with Trasformer """
    def __init__(self, dim=512, n_heads=8, dim_ff=2048, 
                pad_idx=0, vocab_size=700,
                num_basic_block_layer=2,
                num_instruction_layer=2,
                num_op_layer=4):
        super().__init__()

        self.pos_embed = get_positional_encoding_1d(dim)


        self.pad_idx = pad_idx
        self.embed = nn.Embedding(vocab_size, dim, self.pad_idx)


        self.num_basic_block_layer = num_basic_block_layer
        self.num_instruction_layer = num_instruction_layer
        self.num_op_layer = num_op_layer

        self.basic_block = nn.ModuleList(
            [DeepPMTransformerEncoderLayer(dim, n_heads, dim_ff) for _ in range(self.num_basic_block_layer)]
        )

        self.instruction_block = nn.ModuleList(
            [DeepPMTransformerEncoderLayer(dim, n_heads, dim_ff) for _ in range(self.num_instruction_layer)]
        )

        self.op_block = nn.ModuleList(
            [DeepPMTransformerEncoderLayer(dim, n_heads, dim_ff) for _ in range(self.num_op_layer)]
        )

        self.prediction = nn.Linear(dim, 1)

    def _basic_block(self, x, mask):
        batch_size, inst_size, seq_size, _ = x.shape
        x = x.view(batch_size, inst_size * seq_size, -1)
        mask = mask.view(batch_size, inst_size * seq_size)

        for block in self.basic_block:
            x = block(x, mask, use_weighted_attn=False)
        
        x = x.masked_fill(mask.unsqueeze(-1), 0)
        x = x.view(batch_size, inst_size, seq_size, -1)
        return x
    
    def _instruction_block(self, x, mask, op_seq_mask):
        batch_size, inst_size, seq_size, _ = x.shape

        x = x.view(batch_size * inst_size, seq_size, -1)
        mask = mask.view(batch_size * inst_size, seq_size)
        op_seq_mask = op_seq_mask.view(batch_size * inst_size)

        x = x.masked_fill(op_seq_mask.unsqueeze(-1).unsqueeze(-1), 1)
        mod_mask = mask.masked_fill(op_seq_mask.unsqueeze(-1), False)

        for block in self.instruction_block:
            x = block(x, mod_mask, use_weighted_attn=False)
        x = x.masked_fill(mask.unsqueeze(-1), 0)
        x = x.view(batch_size, inst_size, seq_size, -1)
        return x
    
    def _op_block(self, x, op_seq_mask):
        for block in self.op_block:
            x = block(x, op_seq_mask, use_weighted_attn=True)

        x = x.masked_fill(op_seq_mask.unsqueeze(-1), 0)
        return x
       
    def forward(self, x):
        # B I S
        batch_size, inst_size, seq_size = x.shape
        mask = x == self.pad_idx
        op_seq_mask = mask.all(dim=-1)

        #  B I S D
        output = self.embed(x)

        #  B*I S D
        output = output.view(batch_size * inst_size, seq_size, -1)
        output = self.pos_embed(output)

        #  B I S D
        output = output.view(batch_size, inst_size, seq_size, -1)
        output = self._basic_block(output, mask)
        output = self._instruction_block(output, mask, op_seq_mask)

        # reduce
        # B I H
        output = output[:, :, 0]
        output = self.pos_embed(output)

        output = self._op_block(output, op_seq_mask)

        #  B I
        output = output.sum(dim=1)
        output = self.prediction(output).squeeze(1)
        return output
