import torch
import torch.nn as nn
import torch.nn.functional as F

from .pos_encoder import get_positional_encoding_1d
from .base_blocks import Seq, Op, BasicBlock
from .bert import BERT

class BertBaseline(nn.Module):
    """DeepPM model with Trasformer """
    def __init__(self, 
                dim=512, n_heads=8, dim_ff=2048, 
                pad_idx=0, end_idx=9, vocab_size=660, pred_drop=0.0,
                num_basic_block_layer=2,
                num_instruction_layer=2,
                num_op_layer=4, bert_weight=None):
        super().__init__()

        if dim != 512:
             raise ValueError('dim must be 512')
        bert = BERT(vocab_size=660, hidden=512, n_layers=12, attn_heads=8)
        weight = torch.load(bert_weight, map_location=torch.device('cpu'))
        bert.load_state_dict(weight)
        bert.embedding.position = get_positional_encoding_1d(dim)
        bert.eval()

        # freezing bert
        for param in bert.parameters():
            param.requires_grad = False
        print('Loaded pretrained BERT model')

        self.bert = bert

        self.basic_block_layer = BasicBlock(dim, dim_ff, n_heads, num_basic_block_layer, activation='gelu')
        self.instruction_layer = Seq(dim, dim_ff, n_heads, num_instruction_layer, activation='gelu')
        self.op_layer = Op(dim, dim_ff, n_heads, num_op_layer, activation='gelu')


        self.pad_idx = pad_idx
        self.end_idx = end_idx
        self.pos_embed = get_positional_encoding_1d(dim)
        self.prediction = nn.Sequential(
            nn.Dropout(pred_drop),
            nn.Linear(dim, 1)
        )

    def train(self, mode=True):
        super().train(mode)
        self.bert.eval()

    def bert_output_to_deeppm_input(self, x, output):
        all_instrs = []
        all_tensors = []
        
        for x_row, output_row in zip(x, output):
            end_mask = x_row == self.end_idx
            end_idx_tensor = end_mask.nonzero(as_tuple=True)[0]
            lens = end_idx_tensor - torch.cat((torch.tensor([-1], dtype=torch.int, device=x.device), end_idx_tensor[:-1]), dim=0)

            max_seq = max(lens)
            without_padding_len = end_idx_tensor[-1].item() + 1

            x_row = x_row[:without_padding_len]
            instrs = x_row.split(lens.tolist())
            instrs = [F.pad(instr, (0, max_seq-instr.size(0)), value=self.pad_idx) for instr in instrs]
            instrs = torch.stack(instrs)
            all_instrs.append(instrs)

            output_row = output_row[:without_padding_len]
            tensors = output_row.split(lens.tolist())
            tensors = [F.pad(tn, (0, 0, 0, max_seq-tn.size(0)), value=self.pad_idx) for tn in tensors]
            tensors = torch.stack(tensors)
            all_tensors.append(tensors)

        max_instr = max([instr.size(0) for instr in all_instrs])
        max_seq = max([instr.size(1) for instr in all_instrs])
        all_instrs = torch.stack(
            [
                F.pad(instr, (0, max_seq-instr.size(1), 0, max_instr-instr.size(0)), value=self.pad_idx) 
                    for instr in all_instrs
            ]
        )

        all_tensors = torch.stack(
            [
                F.pad(tn, (0, 0, 0, max_seq-tn.size(1), 0, max_instr-tn.size(0)), value=self.pad_idx) 
                    for tn in all_tensors
            ]
        )
        
        return all_instrs, all_tensors

    def forward(self, x):
        output = self.bert(x)
        x, output = self.bert_output_to_deeppm_input(x, output)

        # Basic setup
        batch_size, inst_size, seq_size = x.shape
        mask = x == self.pad_idx
        op_seq_mask = mask.all(dim=-1)
        
        # Instruction layer
        output = self.basic_block_layer(output, mask)
        output = self.instruction_layer(output, mask, op_seq_mask)

        #  Selecting Op
        output = output.sum(dim=2)

        # Op layer
        output = self.pos_embed(output)
        output = self.op_layer(output, op_seq_mask)
        output = output.sum(dim = 1)
        out = self.prediction(output).squeeze(1)

        return out
 