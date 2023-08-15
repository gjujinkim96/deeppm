import torch
import torch.nn as nn
from utils import get_device
from losses import load_losses

from .base_class import BertModule
from .pos_encoder import get_positional_encoding_2d, get_positional_encoding_1d

class MaskedLang(BertModule):
    """DeepPM model with Trasformer """
    def __init__(self, 
                dim=512, n_heads=8, dim_ff=2048, 
                pad_idx=628, vocab_size=700, pred_drop=0.0, drop_p=0.1, unk_idx=5, block_activation='relu',
                num_basic_block_layer=2,
                num_instruction_layer=2,
                num_op_layer=4, loss_type='MapeLoss', loss_fn_arg={}, unk_ratio=0.1, random_ratio=0.1):
        super().__init__()

        self.num_basic_block_layer = num_basic_block_layer
        self.num_instruction_layer = num_instruction_layer
        self.num_op_layer = num_op_layer

        device = get_device(should_print=False)

        if self.num_basic_block_layer > 0:
            block = nn.TransformerEncoderLayer(dim, n_heads, device=device, dim_feedforward=dim_ff, dropout=drop_p,
                                                batch_first=True, activation=block_activation)
            self.basic_block_layer = nn.TransformerEncoder(block, num_basic_block_layer, enable_nested_tensor=False)

        if self.num_instruction_layer > 0:
            block = nn.TransformerEncoderLayer(dim, n_heads, device=device, dim_feedforward=dim_ff, dropout=drop_p,
                                                batch_first=True, activation=block_activation)
            self.instruction_layer = nn.TransformerEncoder(block, num_instruction_layer, enable_nested_tensor=False)

        if self.num_op_layer > 0:
            block = nn.TransformerEncoderLayer(dim, n_heads, device=device, dim_feedforward=dim_ff, dropout=drop_p,
                                                batch_first=True, activation=block_activation)
            self.op_layer = nn.TransformerEncoder(block, num_op_layer, enable_nested_tensor=False)

        self.unk_ratio = unk_ratio
        self.random_ratio = random_ratio
        self.pad_idx = pad_idx
        self.unk_idx = unk_idx
        self.embed = nn.Embedding(vocab_size, dim, padding_idx = pad_idx,
                                dtype=torch.float32, device=device) # token embedding
        self.pos_embed2d = get_positional_encoding_2d(dim)
        self.pos_embed1d = get_positional_encoding_1d(dim)
        self.prediction = nn.Sequential(
            nn.Dropout(pred_drop),
            nn.Linear(dim, 1, dtype=torch.float32)
        )

        self.bert_prediction = nn.Sequential(
            nn.Dropout(pred_drop),
            nn.Linear(dim, vocab_size, dtype=torch.float32)
        )

        self.loss = load_losses(loss_type, loss_fn_arg)
        self.bert_loss = load_losses('CrossEntropyLoss', loss_setting={'ignore_index': self.pad_idx})
       
    def forward(self, x, bert_on=True):
        # Basic setup
        batch_size, inst_size, seq_size = x.shape
        mask = x == self.pad_idx
        op_seq_mask = mask.all(dim=-1)

        if bert_on:
            roll = torch.rand_like(x, dtype=torch.float, requires_grad=False)
            roll = torch.masked_fill(roll, mask, 1.0)
            unk_part = roll < 0.1
            random_part = (0.1 <= roll) & (roll < 0.1 + 0.1)

            x = torch.masked_fill(x, unk_part, 11)

            random_numbers = torch.randint_like(x, 20, 30, requires_grad=False)
            x = torch.masked_fill(x, random_part, 0) + torch.masked_fill(random_numbers, ~random_part, 0)
        
        output = self.embed(x)

        # Adding pos emb
        output = self.pos_embed2d(output)
        output = output.view(batch_size * inst_size, seq_size, -1)

        # Basic block layer
        if self.num_basic_block_layer > 0:
            output = output.view(batch_size, inst_size * seq_size, -1)
            mask = mask.view(batch_size, inst_size * seq_size)
            output = self.basic_block_layer(output, src_key_padding_mask=mask)

        bert_output = self.bert_prediction(output)
        bert_output = bert_output.view(batch_size, inst_size, seq_size, -1)

        # Instruction layer
        if self.num_instruction_layer > 0:
            output = output.view(batch_size * inst_size, seq_size, -1)
            mask = mask.view(batch_size * inst_size, seq_size)
            op_seq_mask = op_seq_mask.view(batch_size * inst_size)

            output = output.masked_fill(op_seq_mask.unsqueeze(-1).unsqueeze(-1), 1)
            mod_mask = mask.masked_fill(op_seq_mask.unsqueeze(-1), False)
            output = self.instruction_layer(output, src_key_padding_mask=mod_mask)
            output = output.masked_fill(op_seq_mask.unsqueeze(-1).unsqueeze(-1), 0)

        #  Selecting Op
        output = output.view(batch_size, inst_size, seq_size, -1)
        op_seq_mask = op_seq_mask.view(batch_size, inst_size)
        output = output[:,:, 1,:]


        # Op layer
        if self.num_op_layer > 0:
            output = self.pos_embed1d(output)
            output = self.op_layer(output, src_key_padding_mask=op_seq_mask)


        output = output.masked_fill(op_seq_mask.unsqueeze(-1), 0)
        output = output.sum(dim = 1)
        out = self.prediction(output).squeeze(1)

        return out, bert_output
    
    def get_loss(self):
        return self.loss, self.bert_loss
    