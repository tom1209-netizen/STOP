# Training the Unified TempMe-STOP Model

This guide explains how to train the unified TempMe-STOP model for temporal video event retrieval.

## Overview

The unified model can be trained in multiple ways:

1. **Joint Training** (Recommended): Train both TempMe and STOP modules together end-to-end
2. **Individual Module Training**: Train only TempMe or only STOP 
3. **Sequential Training**: Train TempMe first, then STOP
4. **Fine-tuning**: Fine-tune pre-trained modules

## Quick Start

### 1. Prepare Data

Ensure your data follows the expected format for your dataset type (MSR-VTT, DiDeMo, etc.):

```bash
# Example for MSR-VTT
data/
├── train/
│   ├── videos/          # Video files
│   └── annotations.json # Text descriptions
├── val/
│   ├── videos/
│   └── annotations.json
└── test/
    ├── videos/
    └── annotations.json
```

### 2. Configure Training

Create or modify the training configuration:

```bash
cp configs/training_config.json configs/my_training_config.json
# Edit the configuration file as needed
```

Key configuration options:

```json
{
  "training": {
    "training_mode": "joint",        // "joint", "tempme_only", "stop_only", "sequential"
    "tempme_lr": 1e-4,              // Learning rate for TempMe module
    "stop_lr": 1e-5,                // Learning rate for STOP module  
    "batch_size": 8,                // Batch size (adjust based on GPU memory)
    "num_epochs": 20,               // Number of training epochs
    "datatype": "msrvtt",           // Dataset type
    "checkpoint_dir": "./checkpoints/unified_model"
  }
}
```

### 3. Start Training

#### Joint Training (Recommended)
Train both modules together for optimal performance:

```bash
python train_unified_model.py \
    --config configs/my_training_config.json \
    --mode joint \
    --seed 42
```

#### Individual Module Training

Train only the TempMe frame compression module:
```bash
python train_unified_model.py \
    --config configs/my_training_config.json \
    --mode tempme_only
```

Train only the STOP retrieval module:
```bash
python train_unified_model.py \
    --config configs/my_training_config.json \
    --mode stop_only
```

#### Sequential Training
Train TempMe first, then STOP:
```bash
python train_unified_model.py \
    --config configs/my_training_config.json \
    --mode sequential
```

### 4. Resume Training

Resume from a checkpoint:

```bash
python train_unified_model.py \
    --config configs/my_training_config.json \
    --mode joint \
    --resume checkpoints/unified_model/checkpoint_epoch_10.pth
```

## Training Strategies

### 1. Joint Training

**Best for**: End-to-end optimization, highest performance

```python
# Configuration
{
  "training": {
    "training_mode": "joint",
    "tempme_lr": 1e-4,
    "stop_lr": 1e-5,
    "num_epochs": 20
  }
}
```

**Advantages**:
- Optimal end-to-end performance
- Shared gradients between modules
- Faster convergence

**Considerations**:
- Requires more GPU memory
- More complex gradient flow

### 2. Sequential Training

**Best for**: Limited GPU memory, stable training

```python
# Automatically trains TempMe first, then STOP
{
  "training": {
    "training_mode": "sequential",
    "num_epochs": 20  // Split between phases
  }
}
```

**Advantages**:
- Lower memory requirements
- Stable training process
- Good for debugging individual modules

### 3. Individual Module Training

**Best for**: Fine-tuning specific components

```bash
# Fine-tune only TempMe with frozen STOP
python train_unified_model.py --mode tempme_only --config my_config.json

# Fine-tune only STOP with frozen TempMe  
python train_unified_model.py --mode stop_only --config my_config.json
```

## Configuration Details

### Core Training Parameters

```json
{
  "training": {
    "batch_size": 8,                    // Adjust based on GPU memory
    "num_epochs": 20,                   // Total training epochs
    "warmup_epochs": 2,                 // Learning rate warmup
    "optimizer": "AdamW",               // "AdamW" or "BertAdam"
    "weight_decay": 0.01,               // L2 regularization
    "gradient_accumulation_steps": 1,   // Effective batch size multiplier
    "max_grad_norm": 1.0,              // Gradient clipping
    "scheduler": "cosine"               // Learning rate schedule
  }
}
```

### Module-Specific Learning Rates

Different learning rates for different components:

```json
{
  "training": {
    "tempme_lr": 1e-4,        // Higher LR for frame compression
    "stop_lr": 1e-5,          // Lower LR for pre-trained STOP
    "tempme_loss_weight": 1.0, // Loss weighting
    "stop_loss_weight": 1.0
  }
}
```

### Data Configuration

```json
{
  "training": {
    "datatype": "msrvtt",              // "msrvtt", "didemo", "activitynet", "vatex"
    "train_data_path": "./data/train",
    "val_data_path": "./data/val"
  }
}
```

## Monitoring Training

### Tensorboard

Monitor training progress:

```bash
tensorboard --logdir checkpoints/unified_model/logs
```

Key metrics to watch:
- Training loss
- Validation loss  
- Learning rates
- Gradient norms

### Checkpoints

Checkpoints are saved automatically:

```
checkpoints/unified_model/
├── logs/                           # Tensorboard logs
├── checkpoint_epoch_2.pth         # Regular checkpoints
├── checkpoint_epoch_4.pth
├── best_model.pth                  # Best performing model
└── config.json                     # Training configuration
```

## Advanced Training

### Custom Loss Functions

Extend the model for custom losses:

```python
class CustomUnifiedModel(UnifiedTempMeSTOPModel):
    def compute_training_loss(self, video_tensor, video_mask, input_ids, attention_mask, token_type_ids):
        # Get base losses
        loss_dict = super().compute_training_loss(video_tensor, video_mask, input_ids, attention_mask, token_type_ids)
        
        # Add custom losses
        loss_dict['custom_loss'] = self.compute_custom_loss(...)
        loss_dict['total_loss'] += loss_dict['custom_loss']
        
        return loss_dict
```

### Multi-GPU Training

For distributed training:

```python
# Add to training script
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

# Initialize distributed training
dist.init_process_group(backend='nccl')
model = DistributedDataParallel(model)
```

### Mixed Precision Training

Already enabled by default with automatic mixed precision (AMP):

```python
# Controlled by scaler in trainer
with torch.cuda.amp.autocast():
    loss = model.compute_training_loss(...)
scaler.scale(loss).backward()
```

## Troubleshooting

### Common Issues

1. **Out of Memory**:
   - Reduce batch_size
   - Increase gradient_accumulation_steps
   - Use sequential training mode

2. **Training Unstable**:
   - Lower learning rates
   - Increase warmup epochs
   - Use gradient clipping

3. **Poor Performance**:
   - Try joint training instead of sequential
   - Adjust loss weights
   - Check data preprocessing

### Performance Tips

1. **Optimal Batch Size**: Find the largest batch size that fits in memory
2. **Learning Rate**: Use different rates for different modules
3. **Gradient Accumulation**: Simulate larger batches if memory limited
4. **Mixed Precision**: Enabled by default for faster training

## Example Training Scripts

### Full Training Pipeline

```python
from unified_model import create_unified_model
from config import UnifiedModelConfig

# Load configuration
config = UnifiedModelConfig.from_json("configs/training_config.json")

# Create and train model
trainer = UnifiedModelTrainer(config)
trainer.train()

# Evaluate
trainer.evaluate(val_dataloader)
```

### Custom Training Loop

```python
# For custom training requirements
model = create_unified_model("configs/training_config.json")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for epoch in range(num_epochs):
    for batch in train_dataloader:
        loss_dict = model.compute_training_loss(*batch)
        loss = loss_dict['total_loss']
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

This comprehensive training system allows you to:
- Train the unified model end-to-end
- Fine-tune individual components
- Experiment with different training strategies
- Monitor and debug training progress
- Scale to multiple GPUs when needed

The training is designed to be flexible and extensible while providing sensible defaults for most use cases.