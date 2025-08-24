# Unified TempMe-STOP Training Guide

This training script (`train_unified_model.py`) follows the same structure as `main.py` and provides comprehensive training capabilities for the unified TempMe-STOP model with all requested features.

## Features

### ✅ Same Structure as main.py
- Uses `get_args()` from `params.py` for argument parsing
- Implements distributed training with `main()` and `main_worker()` functions
- Uses the same imports, utilities, and patterns as `main.py`
- Compatible with existing dataloaders and optimization strategies

### ✅ Retrieval Metrics (R@K)
- Computes R@1, R@5, R@10 retrieval metrics
- Supports both text-to-video and video-to-text retrieval
- Includes Median R and Mean R metrics
- Full compatibility with existing evaluation infrastructure

### ✅ Multi-sentence Evaluation
- Support for datasets with multiple sentences per video
- Proper handling of cut-off points and multi-sentence datasets
- Maintains compatibility with existing multi-sentence evaluation logic

### ✅ CLIP-style logit_scale Clamp
- Implements proper logit_scale clamping: `torch.clamp_(model.clip.logit_scale.data, 0.1, 4.6052)`
- Applied during training loop as in the original CLIP paper
- Prevents logit scale from becoming too large or small

### ✅ Fine-grained Freezing via Name Substring
- New `--freeze_substrings` argument for precise parameter freezing
- New `--keep_substrings` argument for keeping specific parameters trainable
- Enhanced freezing logic with substring matching
- Better control over which parameters to freeze/unfreeze

### ✅ CLIP Backbone Freezing
- Enhanced `--freeze_clip` option with better control
- Ability to freeze entire CLIP backbone while keeping specific modules trainable
- Support for new added modules (LoRA, prompts, etc.)
- Detailed logging of frozen vs trainable parameters

## Usage Examples

### Basic Training
```bash
# Train with all features enabled
python train_unified_model.py \
    --output_dir ./outputs \
    --datatype msrvtt \
    --do_train 1 \
    --epochs 20 \
    --batch_size 32 \
    --lr 1e-4
```

### CLIP Backbone Freezing
```bash
# Freeze entire CLIP backbone, keep only new modules trainable
python train_unified_model.py \
    --output_dir ./outputs \
    --datatype msrvtt \
    --freeze_clip 1 \
    --do_train 1
```

### Fine-grained Freezing
```bash
# Freeze specific components by substring matching
python train_unified_model.py \
    --output_dir ./outputs \
    --datatype msrvtt \
    --freeze_clip 1 \
    --freeze_substrings "visual.transformer" "text.transformer" \
    --keep_substrings "logit_scale" "LoRA" "prompt" \
    --do_train 1
```

### Evaluation with Retrieval Metrics
```bash
# Run evaluation to see R@K metrics
python train_unified_model.py \
    --output_dir ./outputs \
    --datatype msrvtt \
    --do_eval 1 \
    --resume ./outputs/best_model.pth
```

### Multi-sentence Dataset Training
```bash
# Training on datasets with multiple captions per video
python train_unified_model.py \
    --output_dir ./outputs \
    --datatype msrvtt \
    --expand_msrvtt_sentences \
    --do_train 1 \
    --epochs 20
```

## Key Arguments

### Freezing Control
- `--freeze_clip`: Whether to freeze CLIP backbone (0/1)
- `--freeze_substrings`: List of parameter name substrings to freeze
- `--keep_substrings`: List of parameter name substrings to keep trainable (takes priority)

### Training Configuration
- `--optim`: Optimizer choice (BertAdam/AdamW)
- `--lr`: Learning rate
- `--epochs`: Number of training epochs
- `--batch_size`: Training batch size
- `--gradient_accumulation_steps`: Gradient accumulation steps

### Evaluation
- `--do_eval`: Run evaluation only
- `--do_train`: Enable training
- `--datatype`: Dataset type (msrvtt, lsmdc, etc.)

### Distributed Training
- `--gpu`: Specific GPU to use
- `--distributed`: Enable distributed training
- `--world_size`: Number of nodes for distributed training

## Output and Metrics

### Training Logs
- Comprehensive logging with parameter counts
- TensorBoard support for training visualization
- Detailed progress tracking with loss components

### Evaluation Results
```
Text-to-Video:
 (metric) >>>  R@1: 45.2 - R@5: 71.8 - R@10: 81.3 - Median R: 3.0 - Mean R: 12.5
Video-to-Text:
 (metric) >>>  V2T$R@1: 44.1 - V2T$R@5: 70.5 - V2T$R@10: 80.2 - V2T$Median R: 3.0 - V2T$Mean R: 13.1
```

### Checkpoints
- Automatic checkpointing with best model selection
- Resume capability from any checkpoint
- Comprehensive checkpoint information including optimizer state

## Integration with Existing Infrastructure

This training script is fully compatible with:
- All existing dataloaders in `dataloaders/`
- All optimization utilities in `utils/`
- All evaluation metrics in `utils/metrics.py`
- Existing distributed training infrastructure
- TensorBoard logging and visualization

## Advanced Features

### Parameter Group Analysis
- Detailed logging of trainable vs frozen parameters
- Parameter count breakdown by module
- Memory usage tracking

### CLIP-style Training
- Proper logit scale management
- Temperature scaling support
- Mixed precision training support

### Multi-GPU Support
- DistributedDataParallel (DDP) support
- DataParallel (DP) fallback
- Automatic GPU detection and allocation

## Example Training Session

```bash
# Full training session with all features
python train_unified_model.py \
    --output_dir ./unified_training_run \
    --datatype msrvtt \
    --do_train 1 \
    --epochs 30 \
    --batch_size 32 \
    --batch_size_val 64 \
    --lr 1e-4 \
    --freeze_clip 1 \
    --freeze_substrings "visual.ln_" "text.ln_" \
    --keep_substrings "logit_scale" "LoRA" "prompt" "TemporalPrompt" \
    --optim AdamW \
    --warmup_proportion 0.1 \
    --precision amp \
    --gradient_accumulation_steps 2 \
    --n_display 50 \
    --seed 42
```

This will:
1. Train for 30 epochs with mixed precision
2. Freeze CLIP backbone except for specified components
3. Use fine-grained freezing for layer normalization
4. Keep LoRA, prompts, and logit_scale trainable
5. Evaluate with full retrieval metrics (R@1, R@5, R@10)
6. Support multi-sentence evaluation if dataset has it
7. Apply CLIP-style logit_scale clamping
8. Save best model based on R@1 performance