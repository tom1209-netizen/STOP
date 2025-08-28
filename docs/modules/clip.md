# `clip.py`

Port of OpenAI's CLIP model with additional hooks for temporal prompting.

## Components

- **Vision Backbone**
  - `Bottleneck`, `ModifiedResNet` and `AttentionPool2d` implement ResNet-based encoders.
  - `VisualTransformer` provides a Vision Transformer alternative.

- **Text Encoder**
  - `Transformer` composed of `ResidualAttentionBlock` layers with `LayerNorm` and `QuickGELU` activations.

- **Prompt-enabled Blocks**
  - `PromptResidualAttentionBlock` and `PromptTransformer` allow injecting temporal prompts into the Transformer stack.

- **`CLIP` Class**
  - Wraps visual and text encoders, exposing `encode_image`, `encode_text` and `forward` for joint embeddings.
  - Maintains a learnable `logit_scale` to balance similarity scores.

- **Utility Functions**
  - `convert_weights` casts model parameters to `fp16` for efficient inference.
  - `build_clip_model` constructs a CLIP model from a state dictionary.
  - `load_clip_state_dict`, `_download`, and `available_models` manage retrieval of pretrained weights from the internet or cache.

## Role in STOP

`CLIP` serves as the backbone for both modalities in `CLIP4Clip`. Temporal prompting classes interact with `temporal_prompting.py` to integrate frame-wise prompts during video encoding.
