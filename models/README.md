# Utils

## checkpoint_utils.py

- Provides wrapper function for using torch.utils.checkpoint

## pos_encoder.py

- Provides user-friendly wrapper for positional_encodings library

---

# Ithemal 관련

## Ithemal.py

- Implements the Ithemal model.

---

# Bert 관련

## bert_transformer.py

- Implements the transformer used for BERT model.

## bert.py

- Implements the BERT model.

## BertBaseline.py

- Combine BERT with simplified DeepPM model(structure with 2 basic block, 2 seq block, 4 op block + default pytorch Transformer)

---

# 기본 Transformer 사용한 DeepPM 관련

## base_blocks.py

- Contains classes for blocks used by deeppm structure (Seq, BasicBlock, Op)
- Implemented using default pytorch Transformer

---

# DeepPM 관련

## CustomSelfAttention.py

- Implements SelfAttention layer with custom attention mask modifier

## deeppm_transformer.py

- Implements transformer using CustomSelfAttetion 

## deeppm_basic_blocks.py
- Contains classes for blocks used by deeppm structure (Seq, BasicBlock, Op)
- Implemented using transformer from deeppm_transformer.py

## DeepPM.py

- Implements deeppm
