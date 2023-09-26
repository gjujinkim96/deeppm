import torch
import torch.nn as nn

from .pos_encoder import get_positional_encoding_1d
from .deeppm_basic_blocks import DeepPMBasicBlock, DeepPMSeq, DeepPMOp
from utils import get_device

from torch.utils.checkpoint import checkpoint
from utils import get_device
from .checkpoint_utils import method_dummy_wrapper

class WeightEmb(nn.Module):
    def __init__(self, dist_weight_len_limit, dim, use_checkpoint=False, dummy=None):
        super().__init__()

        self.max_vocab = 2*dist_weight_len_limit+2
        self.weight_emb = nn.Embedding(self.max_vocab, dim, self.max_vocab-1)
        self.weight_layer = nn.Sequential(
            nn.Linear(dim + 1, 4 * dim),
            nn.ReLU(),
            nn.Linear(4 * dim, 1),
        )

        self.use_checkpoint = use_checkpoint
        self.dummy = dummy


    def _forward(self, weight, size):
        weight_batch, weight_size, _, = weight.shape

        # vocab
        value = torch.arange(self.max_vocab, device=weight.device)

        # vocab dim
        value = self.weight_emb(value)

        # batch vocab dim
        value = value.unsqueeze(0).expand(weight_batch, -1, -1)

        # size: batch vocab 1
        size = size.view(weight_batch, 1, 1).expand(-1, self.max_vocab, -1)

        # batch vocab dim+1
        value = torch.cat((value, size), dim=-1)

        # batch vocab
        value = self.weight_layer(value).squeeze(-1)

        tmp = []
        for v, w in zip(value, weight):
            tmp.append(torch.index_select(v, 0, w.view(-1)).view(weight_size, weight_size))

        return torch.stack(tmp)

    def forward(self, weight, size):
        if self.use_checkpoint:
            return checkpoint(method_dummy_wrapper(self._forward), self.dummy, weight, size)
        else:
            return self._forward(weight, size)

class DeepPMLearnableWeight(nn.Module):
    """DeepPM model with Trasformer """
    def __init__(self, dim=512, n_heads=8, dim_ff=2048, 
                pad_idx=0, vocab_size=700,
                num_basic_block_layer=2,
                num_instruction_layer=2,
                num_op_layer=4, use_checkpoint=False,
                use_bb_attn=True, use_seq_attn=True, use_op_attn=True, 
                dist_weight_len_limit=40):
        super().__init__()

        self.use_checkpoint = use_checkpoint
        if self.use_checkpoint:
            device = get_device(should_print=False)
            self.dummy = torch.zeros(1, requires_grad=True, device=device)
        else:
            self.dummy = None

        self.pos_embed = get_positional_encoding_1d(dim)


        self.pad_idx = pad_idx
        self.embed = nn.Embedding(vocab_size, dim, self.pad_idx)


        self.weight_layer = WeightEmb(dist_weight_len_limit, 64, 
                                      use_checkpoint=use_checkpoint, dummy=self.dummy)

        self.basic_block = DeepPMBasicBlock(dim, dim_ff, n_heads, num_basic_block_layer, 
                            use_checkpoint=use_checkpoint, dummy=self.dummy)
        self.instruction_block = DeepPMSeq(dim, dim_ff, n_heads, num_instruction_layer, 
                            use_checkpoint=use_checkpoint, dummy=self.dummy)
        self.op_block = DeepPMOp(dim, dim_ff, n_heads, num_op_layer, 
                            use_checkpoint=use_checkpoint, dummy=self.dummy)

        self.prediction = nn.Linear(dim, 1)

        self.use_bb_attn = use_bb_attn
        self.use_seq_attn = use_seq_attn
        self.use_op_attn = use_op_attn
       
    def forward(self, x):
        bb_attn_mod = x['bb_attn_mod'] if self.use_bb_attn else None
        bb_sizes = x['bb_sizes'] if self.use_bb_attn else None

        seq_attn_mod = x['seq_attn_mod'] if self.use_seq_attn else None
        seq_sizes = x['seq_sizes'] if self.use_seq_attn else None

        op_attn_mod = x['op_attn_mod'] if self.use_op_attn else None
        op_sizes = x['op_sizes'] if self.use_op_attn else None

        x = x['x']

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

        bb_attn_mod = self.weight_layer(bb_attn_mod, bb_sizes)
        output = self.basic_block(output, mask, bb_attn_mod)

        seq_attn_mod = self.weight_layer(seq_attn_mod, seq_sizes)
        output = self.instruction_block(output, mask, op_seq_mask, seq_attn_mod)

        # reduce
        # B I H
        output = output[:, :, 0]
        output = self.pos_embed(output)

        op_attn_mod = self.weight_layer(op_attn_mod, op_sizes)
        output = self.op_block(output, op_seq_mask, op_attn_mod)

        #  B I
        output = output.sum(dim=1)
        output = self.prediction(output).squeeze(1)
        return output
