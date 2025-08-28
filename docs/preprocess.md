# Preprocessing Utilities

Scripts for preparing raw video data and annotations prior to training.

- **`check_video.py`**: verifies video integrity and logs problematic files.
- **`compress_video.py`**: rescales and compresses videos to the target resolution and bitrate.
- **`folder2lmdb.py`**: converts a directory of video files into an LMDB database for efficient random access.
- **`generate_video_path.py`**: builds `video_path.json` mappings from video identifiers to file locations.
- **`patch_video.py`**: applies spatial patches or crops to video frames for data augmentation.
- **`visualize_video.py`**: utility for rendering sampled frames to check preprocessing results.

These tools operate independently of the training code but ensure data conforms to the expected format consumed by loaders in `dataloaders/`.
