# Model Modules

Detailed documentation for the core modules that implement the STOP model. The files below correspond to code in the `modules/` directory and describe their purpose, key classes or functions, inputs and outputs, and how they interact with the rest of the system.

- [`__init__.py`](init.md) – exposes the primary public interfaces such as `CLIP4Clip`, `SimpleTokenizer`, and `convert_weights`.
- [`base.py`](base.md) – configuration management and `PreTrainedModel` utilities for weight loading and initialisation.
- [`clip.py`](clip.md) – adapts OpenAI CLIP with temporal prompting hooks for video encoding.
- [`clip4clip.py`](clip4clip.md) – main STOP retrieval model combining CLIP encoders with cross-modal interaction and temporal modelling.
- [`module_cross.py`](module_cross.md) – cross-modal Transformer that fuses video and text representations.
- [`losses.py`](losses.md) – training objectives including cross-entropy and ranking losses.
- [`file.py`](file.md) – cache-aware resource loader for HTTP and S3 sources.
- [`simple_tokenizer.py`](simple_tokenizer.md) – byte-pair encoding tokenizer used by the text encoder.
- [`temporal_prompting.py`](temporal_prompting.md) – constructs temporal prompts that modulate frame features.
- [`utils.py`](utils.md) – distributed training helpers and attribute utilities.
- [`cross-base/cross_config.json`](cross_config.md) – default hyperparameters for the cross-modal Transformer.
