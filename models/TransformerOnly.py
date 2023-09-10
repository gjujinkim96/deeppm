import torch.nn as nn

from .pos_encoder import get_positional_encoding_2d
from .base_blocks import Seq, Op, BasicBlock

class TransformerOnly(nn.Module):
    def __init__(self,
                dim=512, n_heads=8, dim_ff=2048, 
                pad_idx=628, vocab_size=700, pred_drop=0.0,
                num_basic_block_layer=8, use_checkpoint=False):
        super().__init__()

        self.use_checkpoint = use_checkpoint

        self.basic_block = BasicBlock(dim, dim_ff, n_heads, num_basic_block_layer, use_checkpoint=use_checkpoint)

        self.pad_idx = pad_idx
        self.embed = nn.Embedding(vocab_size, dim, padding_idx = pad_idx)
        self.pos_embed_2d = get_positional_encoding_2d(dim)
        self.prediction = nn.Sequential(
            nn.Dropout(pred_drop),
            nn.Linear(dim, 1)
        )

    def forward(self, x):
        batch_size, inst_size, seq_size = x.shape
        mask = x == self.pad_idx
        op_seq_mask = mask.all(dim=-1)

        # Embbedding
        # B I S H
        output = self.embed(x)
        output = self.pos_embed_2d(output)
        output = self.basic_block(output, mask)

        # Reduction
        # B I H
        output = output.sum(dim=2)
        
        # B I
        output = self.prediction(output).squeeze(2)

        # B
        output = output.sum(dim=1)

        return output
    