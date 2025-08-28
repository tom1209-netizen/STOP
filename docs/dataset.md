# Dataset Annotations

Sample annotation files used for experimentation and unit tests. These files provide minimal subsets of larger benchmarks.

## ActivityNet
- `train.json`, `val_1.json`: caption annotations for training and validation.
- `train_ids.json`, `val_ids.json`: lists of video identifiers.
- `video_path.json`: mapping from video identifiers to file system paths.
- `train_list.txt`, `val_1_list.txt`: text lists of video files for preprocessing.

## LSMDC
- `LSMDC16_challenge_1000_publictect.csv`: evaluation CSV from the LSMDC challenge.

## MSVD
- `train_list.txt`, `val_list.txt`, `test_list.txt`: video identifier splits for the MSVD dataset.

These annotations are consumed by the loaders in `dataloaders/` and by preprocessing scripts to locate and parse video-caption pairs.
