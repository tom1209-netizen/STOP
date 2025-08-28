# `losses.py`

Collection of training objectives used for contrastive video–text retrieval.

## Loss Functions

- **`CrossEn`**
  - Computes cross-entropy over similarity matrices by applying `log_softmax` along the text dimension and averaging the negative log-diagonal.

- **`MILNCELoss`**
  - Implements the Multiple Instance Learning variant of InfoNCE.
  - Uses a block mask to handle multiple positive pairs per sample and aggregates negatives from both modalities.

- **`MaxMarginRankingLoss`**
  - Margin-based ranking objective with optional negative weighting.
  - Supports hard/easy negative sampling via `hard_negative_rate` and scales losses when multiple pairs are present.

## Role in STOP

`CLIP4Clip` primarily employs `CrossEn` but alternative losses can be used for experimentation with different retrieval criteria.
