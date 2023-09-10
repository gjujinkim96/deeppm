import torch
import torch.nn as nn
import math

from .bert_transformer import TransformerBlock

class TokenEmbedding(nn.Embedding):     #토큰을 임베딩 벡터로 변환
    def __init__(self, vocab_size, embed_size=512):
        super().__init__(vocab_size, embed_size, padding_idx=0)     #embed_size: 임베딩 벡터의 차원 결정, padding_idx: 패딩토큰 인덱스 지정
        #print("TokenEmbedding vocab_size:", vocab_size)  # vocab_size 출력

class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=None):
        super().__init__()

        if max_len is None:
            max_len = 512

        print(d_model)  #999
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()  #pe: 위치 임베딩 행렬. max_len x d_model 크기고 0으로 초기화됨
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)    #0부터 max_len까지의 값 가지는 텐서
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()     #각 차원에 따른 감소율. 위치랑 차원에 따라 다른 주기 ?

        #pe 각각 shape 보기
        #print('pe---------------------------')
        #print(pe)
        #print(pe.shape) #torch.size([1000, 864])
        pe[:, 0::2] = torch.sin(position * div_term)    #짝수 인덱스는 sin 함수로 위치 임베딩 계산
        #print('pe-sin ------------------------------')
        #print(pe)
        #print(pe.size)
        #print(pe.shape) #torch.size([1000, 864])
        pe[:, 1::2] = torch.cos(position * div_term)    #홀수 인덱스는 cos 함수로 위치 임베딩 계산
        #print('pe-cos ------------------------------')
        #print(pe)
        #print(pe.size)
        #print(pe.shape)
        #PE(pos, 2i)     = sin(pos / 10000^(2i/d_model))
        #PE(pos, 2i + 1) = cos(pos / 10000^(2i/d_model))


        pe = pe.unsqueeze(0)    #배치 추가
        self.register_buffer('pe', pe)

    def forward(self, x):
        pe = self.pe.repeat(x.size(0), 1, 1)
        return pe[:, :x.size(1)]
    
class BERTEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)

        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, vocab_size, embed_size, dropout=0.1):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        # self.position = PositionalEncoding1D(embed_size)
        self.position = PositionalEmbedding(d_model=self.token.embedding_dim)
        #self.segment = SegmentEmbedding(embed_size=self.token.embedding_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

    def forward(self, sequence):
        #print('--------BERTEmbedding sequence-----------')
        #print(sequence)     #bert_input [1,150]으로 들어옴
        token_embeddings = self.token(sequence)
        #print('token_embeddings-------------------------')
        #print(token_embeddings)
        #print(token_embeddings.shape)   #[1,150,864]
        # positional_embeddings = self.position(sequence)

        x = self.position(token_embeddings)
        return self.dropout(x)


class BERT(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, vocab_size, hidden=512, n_layers=12, attn_heads=8, dropout=0.1):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden * 4

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=hidden)    #embed_size 768

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)])
 
    def forward(self, x):
        # attention masking for padded token
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
        #mask = (x != 1111).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)    #패딩 인덱스 1111로 바꿈
        #print('-------------x before mask')
        #print(x)
        #print(x.shape)  #[1,2,15]
        ##############mask = (x != 0).unsqueeze(1).repeat(1, 1, x.size(1), 1)
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        #print('----------mask------------')
        #print(mask)         #마스크 어텐션 아래의 shape로 만들어짐
        #print(mask.shape)   #[8, 1,150,150]
        #print('---------------------------')

        #print('--------------x before embedding-------------')
        #print(x)        #bert_input(basic block 1개), [1,2,15]형태
        ##########################x = x.reshape(-1, x.shape[-1])
        #print(x)        #위에꺼에서 괄호 없앰
        #print(x.shape)  #[2,15]로 바뀜
        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x)
        ##print('embedded x---------------------')
        #print(x)
        #print(x.shape)  #torch.Size([1, 150, 864])

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)        #x[8, 150, 512], mask=[1,1,4,15]

        return x
