# `clip4clip.py`

Implements the STOP video–text retrieval model by extending CLIP to handle temporal sequences and cross-modal reasoning.

## Key Classes

- **`CLIP4ClipPreTrainedModel`**
  - Inherits from `PreTrainedModel` and encapsulates both CLIP and cross-modal components.
  - The `from_pretrained` class method loads a pretrained CLIP backbone, initialises cross-modal parameters and applies optional temperature scaling.

- **`TemporalModelling`**
  - Stack of `ResidualAttentionBlock` layers that aggregates per-frame features.
  - Configurable via width, number of layers and attention heads.

- **`CLIP4Clip`**
  - Combines CLIP image/text encoders with a `CrossModel` from [`module_cross.py`](module_cross.md).
  - Supports multiple similarity headers (e.g., `meanP`, `seqLSTM`, `tightTransf`) and temporal prompt injection through `get_TemporalPrompt`.
  - Computes video–text similarities and applies the `CrossEn` loss during training.
  - Utilises distributed helpers `all_gather`, `log_info` and `update_attr` from [`utils.py`](utils.md).

## Supporting Layers

Auxiliary definitions such as `QuickGELU`, `LayerNorm`, and `ResidualAttentionBlock` enable efficient Transformer-style computations.

## Interaction

`CLIP4Clip` is the primary model trained and evaluated in STOP. It orchestrates frame encoding, temporal modelling, cross-modal fusion and loss computation to produce retrieval logits.
