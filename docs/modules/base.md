# `base.py`

Infrastructure for configuration management and a base class for pretrained models.

## Key Classes

- **`PretrainedConfig`**
  - Loads configuration and associated weights from local files or URLs via `get_config`.
  - Serialises and deserialises settings through `from_json_file`, `from_dict`, `to_dict`, and `to_json_string`.
  - Tracks archive map names (`config_name`, `weights_name`) used by `CrossModel`.

- **`PreTrainedModel`**
  - Parent for all learnable modules requiring pretrained weights.
  - Handles parameter initialisation (`init_weights`) and weight loading (`init_preweight`).
  - Exposes abstract `resize_token_embeddings` for token embedding adjustment.

## Activation Helpers

Defines common activation functions `gelu`, `swish`, and a lookup table `ACT2FN` so modules can reference them consistently.

## Interaction Within STOP

`CLIP4Clip`, `CrossModel`, and other models inherit from `PreTrainedModel` to share loading utilities and weight initialisation. Configuration objects derived from `PretrainedConfig` drive the construction of these models.
