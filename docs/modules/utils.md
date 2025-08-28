# `utils.py`

Miscellaneous helper functions used across the STOP codebase.

## Logging and Configuration

- **`log_info`** – rank-aware logger that prints messages only from the main distributed process.
- **`update_attr`** – copies attributes from one configuration object to another, respecting default values.

## Distributed Utilities

- **`AllGather` autograd function** – gathers tensors from all workers while retaining gradients for backpropagation.
- **`all_gather`** – convenience wrapper that returns the concatenated tensor across distributed processes.

These functions support multi-GPU training and are used extensively in [`clip4clip.py`](clip4clip.md) for contrastive learning with large negative pools.
