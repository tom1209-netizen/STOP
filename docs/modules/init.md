# `__init__.py`

Provides convenient, package-level imports for frequently used components. Importing `modules` exposes:

- **`CLIP4Clip`** – main video–text retrieval model defined in [`clip4clip.py`](clip4clip.md).
- **`SimpleTokenizer`** – byte‑pair encoding tokenizer from [`simple_tokenizer.py`](simple_tokenizer.md).
- **`convert_weights`** – utility function from [`clip.py`](clip.md) that converts model parameters to half precision.

These symbols allow users to construct models and tokenizers without referencing deep module paths.
