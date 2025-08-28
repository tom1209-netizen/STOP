# Scripts and Execution

Shell scripts and utilities to reproduce experiments.

- **`scripts/msrvtt.sh`**: example command line for training on the MSRVTT dataset.
- **`scripts/activitynet.sh`**: training script for ActivityNet retrieval.
- **`scripts/vatex.sh`**: training configuration for VATEX.
- **`kill.sh`**: helper to terminate residual training processes.
- **`search_for_best_r1_with_qb_norm.py`**: evaluates different QB normalization strategies to maximise R@1.

These scripts rely on `main.py` and `params.py` and provide ready-to-use experiment setups.
