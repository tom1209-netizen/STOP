# `module_cross.py`

Defines the cross-modal Transformer used to fuse visual and textual embeddings.

## Configuration

- **`CrossConfig`** extends `PretrainedConfig` and stores hyperparameters such as hidden size, number of layers, attention heads and dropout rates. Configurations are typically loaded from [`cross_config.json`](cross_config.md).

## Model Architecture

- **`CrossEmbeddings`** adds positional encodings to concatenated video and text features.
- **`ResidualAttentionBlock`** and **`Transformer`** implement multi-head self-attention over the joint sequence.
- **`CrossPooler`** normalises and projects the `[CLS]` token representation.
- **`CrossModel`** wraps embeddings, transformer and pooler, and exposes a forward pass that returns pooled video–text features.

## Usage in STOP

`CLIP4Clip` instantiates `CrossModel` to compute interaction-aware representations before similarity scoring. The configuration and parameter initialisation leverage `PreTrainedModel` utilities from [`base.py`](base.md).
