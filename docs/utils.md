# Utility Helpers

Supporting modules used across training and evaluation.

- **`dist_utils.py`**: wrappers for PyTorch distributed utilities (`init_distributed_mode`, `get_rank`, `is_master`).
- **`log.py`**: logging configuration with queue-based handlers for multi-process setups.
- **`lr_scheduler.py`**: cosine learning-rate scheduler with warmup (`lr_scheduler` function).
- **`metrics.py`**: retrieval metric computation including recall at K and mean rank.
- **`misc.py`**: miscellaneous helpers such as random seed setting, model checkpoint saving, and float precision conversions.
- **`optimization.py`**: optimiser utilities, notably `BertAdam` and helper `prep_optim_params_groups` for parameter-specific learning rates.
