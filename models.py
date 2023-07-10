# Copyright 2018 Dong-Hyun Lee, Kakao Brain.
# (Strongly inspired by original Google BERT code and Hugging Face's code)

""" Transformer Model Classes & Config Class """

import math
import json
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import split_last, merge_last, get_device


class Config(NamedTuple):
    "Configuration for BERT model"
    vocab_size: int = None # Size of Vocabulary
    dim: int = 768 # Dimension of Hidden Layer in Transformer Encoder
    n_layers: int = 12 # Numher of Hidden Layers
    n_heads: int = 12 # Numher of Heads in Multi-Headed Attention Layers
    dim_ff: int = 768*4 # Dimension of Intermediate Layers in Positionwise Feedforward Net
    #activ_fn: str = "gelu" # Non-linear Activation Function Type in Hidden Layers
    p_drop_hidden: float = 0.1 # Probability of Dropout of various Hidden Layers
    p_drop_attn: float = 0.1 # Probability of Dropout of Attention Layers
    max_len: int = 32 # Maximum Length for Positional Embeddings
    #n_segments: int = 2 # Number of Sentence Segments
    pad_idx: int = 628#500#230

    @classmethod
    def from_json(cls, file):
        return cls(**json.load(open(file, "r")))

    def set_vocab_size(cls, size):
        Config.vocab_size = size

    def set_pad_idx(cls, pad_idx):
        Config.pad_idx = pad_idx


def gelu(x):
    "Implementation of the gelu activation function by Hugging Face"
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class LayerNorm(nn.Module):
    "A layernorm module in the TF style (epsilon inside the square root)."
    def __init__(self, cfg, variance_epsilon=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(cfg.dim))
        self.beta  = nn.Parameter(torch.zeros(cfg.dim))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta

class PositionalEncoding(nn.Module):

        def __init__(self, d_hid, n_position=256):
            super(PositionalEncoding, self).__init__()

            self.n_position = n_position
            # Not a parameter
            self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

        def _get_sinusoid_encoding_table(self, n_position, d_hid):
            ''' Sinusoid position encoding table '''
            def get_position_angle_vec(position):
                return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

            sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
            sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
            sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
            # print(torch.tensor(sinusoid_table).unsqueeze(0).shape)
            return torch.tensor(sinusoid_table).unsqueeze(0).to(dtype=torch.float32)

        def forward(self, x):
            # print('x', x.shape)
            max_size = min(x.size(1), self.n_position)
            return x + self.pos_table[:, :max_size]


class Embeddings(nn.Module):
    "The embedding module from word, position and token_type embeddings."
    def __init__(self, cfg):
        super().__init__()
        self.tok_embed = nn.Embedding(cfg.vocab_size+1, cfg.dim, padding_idx = cfg.pad_idx) # token embedding
        #self.pos_embed = nn.Embedding(32, cfg.dim) # position embedding

        # drop
        #self.norm = nn.LayerNorm(cfg.dim)#LayerNorm(cfg)
        #self.drop = nn.Dropout(cfg.p_drop_hidden)


    def forward(self, x):
        #seq_len = x.size(1)
        #pos = torch.arange(seq_len, dtype=torch.long, device=x.device)

        e = self.tok_embed(x) #+ self.pos_embed(pos)
        return e 
        
        #drop
        #return self.drop(self.norm(e))


class MultiHeadedSelfAttention(nn.Module):
    """ Multi-Headed Dot Product Attention """
    def __init__(self, cfg):
        super().__init__()
        self.proj_q = nn.Linear(cfg.dim, cfg.dim)
        self.proj_k = nn.Linear(cfg.dim, cfg.dim)
        self.proj_v = nn.Linear(cfg.dim, cfg.dim)
        self.scores = None # for visualization
        self.n_heads = cfg.n_heads

        self.output = nn.Linear(cfg.dim, cfg.dim)
        #drop
        #self.drop = nn.Dropout(cfg.p_drop_attn)

    def forward(self, x):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        mask : (B(batch_size) x S(seq_len))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2)
                   for x in [q, k, v])
        del x
        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
 
        del q
        del k

        # masking
        #b,h,s,s = scores.size()
        #indices = torch.triu_indices(s,s,offset=1)
        #scores[:,:,indices[0],indices[1]] = float('-inf')

        b,h,s,s = scores.size()
        masking = np.zeros((s,s))
        for i in range(s):
            for j in range(s):
                if i<j: masking[i][j] = (s+i-j)/s
                else: masking[i][j] = (s-i+j)/s
        masking = torch.Tensor(masking).cuda()
        scores = scores * masking

        del masking

        scores = F.softmax(scores, dim=-1)

        #drop
        #scores = self.drop(F.softmax(scores, dim=-1))

        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        h = (scores @ v).transpose(1, 2).contiguous()
        # -merge-> (B, S, D)
        h = merge_last(h, 2)
        self.scores = scores

        del v
        del scores


        return self.output(h)


class PositionWiseFeedForward(nn.Module):
    """ FeedForward Neural Networks for each position """
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg.dim, cfg.dim_ff)
        self.fc2 = nn.Linear(cfg.dim_ff, cfg.dim)
        #self.activ = lambda x: activ_fn(cfg.activ_fn, x)

    def forward(self, x):
        # (B, S, D) -> (B, S, D_ff) -> (B, S, D)
        return self.fc2(gelu(self.fc1(x)))


class Block(nn.Module):
    """ Transformer Block """
    def __init__(self, cfg):
        super().__init__()
        self.attn = MultiHeadedSelfAttention(cfg)
        self.proj = nn.Linear(cfg.dim, cfg.dim)
        #self.norm1 =  nn.LayerNorm(cfg.dim)#LayerNorm(cfg)
        self.pwff = PositionWiseFeedForward(cfg)
        #self.norm2 =  nn.LayerNorm(cfg.dim)#LayerNorm(cfg)

        #drop
        #self.drop = nn.Dropout(cfg.p_drop_hidden)

    def forward(self, x):
        h = self.attn(x)
        
        #drop
        #h = self.norm1(x + self.drop(self.proj(h)))
        #h = self.norm2(h + self.drop(self.pwff(h)))

        #h = self.norm1(x + self.proj(h))
        #h = self.norm2(h + self.pwff(h))
        h = x + self.proj(h)
        h = h + self.pwff(h)
        return h


class Transformer(nn.Module):
    """ Transformer with Self-Attentive Blocks """
    def __init__(self, cfg):
        super().__init__()
        self.embed = Embeddings(cfg)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])

    def forward(self, x, seg, mask):
        h = self.embed(x, seg)
        for block in self.blocks:
            h = block(h, mask)
        return h

class DeepPM(nn.Module):
    """DeepPM model with Trasformer """
    def __init__(self, cfg):
        super().__init__()


        block = nn.TransformerEncoderLayer(cfg.dim, cfg.n_heads, dtype=torch.float32, device=get_device(),
                                            batch_first=True)
        self.blocks = nn.TransformerEncoder(block, cfg.n_layers)

        self.pad_idx = cfg.pad_idx
        self.embed = nn.Embedding(cfg.vocab_size+1, cfg.dim, padding_idx = cfg.pad_idx,
                                dtype=torch.float32, device=get_device()) # token embedding

        # self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])#cfg.n_layers)])
        # self.pre_blocks = nn.ModuleList([Block(cfg) for _ in range(2)])#cfg.n_layers)])
        # self.token_blocks = nn.ModuleList([Block(cfg) for _ in range(2)])#cfg.n_layers)])

        self.pos_embed = PositionalEncoding(cfg.dim, cfg.max_len).to(get_device())
        self.max_len = cfg.max_len
        #self.pos_embed = nn.Embedding(250, cfg.dim) # position embedding, 1500
        # self.instruction_blocks = nn.ModuleList([Block(cfg) for _ in range(4)])#cfg.n_layers)])
        self.prediction = nn.Linear(cfg.dim, 1, dtype=torch.float32)
       
    def forward(self, item):
        # print('item device', item.device) 
        inputs = item[:, :self.max_len]
        padding_mask = (inputs == self.pad_idx)
        t_output = self.embed(inputs)
        # print(t_output.size())
        t_output = self.pos_embed(t_output)
        # print(t_output.size())

        # batch_size, n_instr, n_dim = t_output.size()

        # B, L, D
        # print(t_output.dtype)
        t_output = self.blocks(t_output, src_key_padding_mask=padding_mask)
    

        # t_output = t_output.view([batch_size, n_instr, n_token, n_dim])

        t_output = t_output[:, 0, :]
        out = self.prediction(t_output).squeeze(1)

        return out
            

