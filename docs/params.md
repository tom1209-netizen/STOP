# Hyperparameters

## `params.py`
Centralised configuration for experiments.

- `get_default_params(model_name)`: returns recommended optimiser defaults for a given CLIP backbone.
- `get_args(description)`: defines the full command line interface for training and evaluation. Parameters cover dataset paths, optimisation settings, distributed training, prompt design, and precision options. Returns a populated `argparse.Namespace` and ensures output directories exist.
- `save_hp_to_json(directory, args)`: helper to persist hyperparameters to `hparams_train.json` for reproducibility.

The module is consumed by `main.py` to configure runs and by preprocessing scripts that require consistent hyperparameters.
