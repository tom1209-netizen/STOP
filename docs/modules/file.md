# `file.py`

Utility functions for locating, downloading and caching external resources such as pretrained weights.

## Highlights

- **Caching Helpers**
  - `url_to_filename` and `filename_to_url` map URLs to deterministic cache filenames.
  - `cached_path` resolves a local path or downloads a remote file (HTTP/S3), storing metadata for future reuse.

- **S3/HTTP Interfaces**
  - `split_s3_path`, `s3_etag`, `s3_get` and `http_get` handle platform-specific downloads.
  - `get_from_cache` orchestrates retrieval using ETag-based caching and progress bars via `tqdm`.

These utilities are leveraged by configuration loaders in [`base.py`](base.md) and model builders like [`clip.py`](clip.md) to transparently fetch pretrained checkpoints.
