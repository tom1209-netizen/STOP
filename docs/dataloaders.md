# Data Loaders

This directory provides dataset interfaces and video decoding utilities.

## Retrieval Datasets
- **`data_dataloaders.py`**: registry mapping dataset names to loader constructors. Exposes `dataloader_msrvtt_train` and `dataloader_msrvtt_test` used by `main.py`.
- **`dataloader_msrvtt_retrieval.py`**: training and evaluation loaders for the MSRVTT captioning dataset. Includes `MSRVTT_DataLoader` and `MSRVTT_TrainDataLoader` classes handling tokenisation, frame sampling and video decoding.
- **`dataloader_activitynet_retrieval.py`**: loader for ActivityNet text-video retrieval. Builds caption dictionaries and sampling of temporal segments.
- **`dataloader_didemo_retrieval.py`**: dataset wrapper for the DiDeMo benchmark with support for multiple captions per video clip.
- **`dataloader_vatex_retrieval.py`**: implements VATEX training and validation loaders.

## Video Decoding and Augmentation
- **`decode.py`**: `RawVideoExtractorpyAV` uses PyAV to sample and transform frames with optional LMDB-backed storage.
- **`rawvideo_util.py`**: alternative frame extraction using OpenCV; exposes `RawVideoExtractorCV2` and helper functions for frame ordering.
- **`sampling.py`**: uniform and multi-segment temporal sampling strategies.
- **`transforms.py`**: tensor-based normalisation, cropping and conversion utilities used during decoding.
