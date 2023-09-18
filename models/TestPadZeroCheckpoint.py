import torch.nn as nn

from .pos_encoder import get_positional_encoding_1d
from .base_blocks_varient import CustomSeq, CustomOp, CustomBasicBlock

class TestPadZeroCheckpoint(nn.Module):
    def __init__(self,
                dim=512, n_heads=8, dim_ff=2048, 
                pad_idx=628, vocab_size=700, pred_drop=0.0,
                num_basic_block_layer=2,
                num_instruction_layer=2,
                num_op_layer=4, use_checkpoint=False):
        super().__init__()

        self.use_checkpoint = use_checkpoint

        self.basic_block = CustomBasicBlock(dim, dim_ff, n_heads, num_basic_block_layer, use_layernorm=False,
                                    use_checkpoint=use_checkpoint)
        self.instruction_block = CustomSeq(dim, dim_ff, n_heads, num_instruction_layer, use_layernorm=False,
                                    use_checkpoint=use_checkpoint)
        self.op_block = CustomOp(dim, dim_ff, n_heads, num_op_layer, use_layernorm=False,
                                    use_checkpoint=use_checkpoint)

        self.pad_idx = pad_idx
        self.embed = nn.Embedding(vocab_size, dim, padding_idx = pad_idx)
        self.pos_embed = get_positional_encoding_1d(dim)
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

        # B*I S H
        output = output.view(batch_size * inst_size, seq_size, -1)
        output = self.pos_embed(output)

        # B I S H
        output = output.view(batch_size, inst_size, seq_size, -1)

        output = self.basic_block(output, mask)
        output = self.instruction_block(output, mask, op_seq_mask)

        # Reduction
        # B I H
        output = output.sum(dim=2)

        output = self.op_block(output, op_seq_mask)
        
        # B I
        output = self.prediction(output).squeeze(2)

        # B
        output = output.sum(dim=1)

        return output
    