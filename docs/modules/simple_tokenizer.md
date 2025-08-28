# `simple_tokenizer.py`

Lightweight byte‑pair encoding (BPE) tokenizer compatible with CLIP.

## Features

- Loads merges from `bpe_simple_vocab_16e6.txt` (included in the repository) using cached helpers.
- Implements text normalisation via `basic_clean` and `whitespace_clean`.
- Provides `encode`, `decode`, `tokenize`, and `convert_tokens_to_ids` methods for preprocessing captions.
- Uses `bytes_to_unicode` mappings to ensure reversible encoding.

## Usage

The tokenizer is constructed implicitly when `SimpleTokenizer` is imported through [`__init__.py`](init.md). It converts raw strings into integer token IDs consumed by the CLIP text encoder.
