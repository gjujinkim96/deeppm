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
