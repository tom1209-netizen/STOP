# `temporal_prompting.py`

Generates learnable prompts that modulate frame features before encoding.

## API

- **`get_TemporalPrompt(args)`** – factory function returning a prompt module based on `args.temporal_prompt`.
- **`TemporalPrompt_3`**
  - Applies a series of 3D convolutions followed by an MLP to produce per-frame prompt tokens.
  - `get_mask` computes saliency masks via a dedicated convolutional network.
  - `init_InterFramePrompt` and `get_inter_frame_prompt` introduce inter-frame attention for enhanced temporal context.

## Interaction

`clip4clip.py` calls `get_TemporalPrompt` to attach prompt generators that inject additional tokens or masks into the CLIP visual encoder, enabling temporal adaptation.
