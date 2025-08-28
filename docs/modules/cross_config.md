# `cross-base/cross_config.json`

Default configuration file for the cross-modal Transformer.

| Field | Description |
|-------|-------------|
| `hidden_size` | Dimensionality of token embeddings (512). |
| `num_hidden_layers` | Number of Transformer layers (4). |
| `num_attention_heads` | Multi-head attention heads (8). |
| `intermediate_size` | Feed-forward layer size (2048). |
| `hidden_act` | Activation function (`gelu`). |
| `hidden_dropout_prob` | Dropout applied to fully connected layers (0.1). |
| `attention_probs_dropout_prob` | Dropout on attention weights (0.1). |
| `max_position_embeddings` | Maximum sequence length (77). |
| `vocab_size` | Token vocabulary size for joint inputs (512). |
| `initializer_range` | Standard deviation for parameter initialisation (0.02). |

`CrossConfig` in [`module_cross.py`](module_cross.md) loads these parameters when constructing the `CrossModel`.
