# Main Application

## `main.py`
Implements the full training and evaluation loop for STOP. Major responsibilities include:

- Parsing configuration through `params.get_args`.
- Initialising distributed training and logging utilities.
- Constructing the `CLIP4Clip` model with optional frozen backbone and temporal prompting modules.
- Building dataset loaders via `dataloaders.data_dataloaders.DATALOADER_DICT`.
- Running training epochs with `train_epoch` and evaluation with `eval_epoch`.
- Saving checkpoints using `utils.misc.save_checkpoint` and reporting retrieval metrics from `utils.metrics`.

Key entry points:

- `main(args)`: orchestrates setup and spawns worker processes for distributed execution.
- `main_worker(gpu, ngpus_per_node, log_queue, args)`: configures the model, optimisers, and dataloaders.
- `train_epoch(...)`: performs one optimisation epoch with optional mixed precision.
- `eval_epoch(...)`: caches text and video features then computes similarity scores.
- `_run_on_single_gpu(...)`: helper for similarity computation between cached text and video representations.

The script interacts extensively with modules under `modules/`, dataset loaders under `dataloaders/`, and utilities from `utils/` for logging, optimisation, and distributed training.
