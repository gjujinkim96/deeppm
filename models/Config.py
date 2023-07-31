from typing import NamedTuple
import json

class Config(NamedTuple):
    "Configuration for BERT model"
    model_class: str = None
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
    pred_drop: float = 0
    stacked: bool = False
    only_unique: bool = False

    @classmethod
    def from_json(cls, file):
        return cls(**json.load(open(file, "r")))

    def set_vocab_size(cls, size):
        Config.vocab_size = size

    def set_pad_idx(cls, pad_idx):
        Config.pad_idx = pad_idx

    def set_model_class(cls, model_class):
        Config.model_class = model_class
