#!/bin/bash

# TempMe-STOP Training Script
# This script trains the integrated TempMe-STOP model

# Set paths (modify these for your setup)
DATA_DIR="/path/to/your/data"
OUTPUT_DIR="outputs/tempme_stop_experiment_$(date +%Y%m%d_%H%M%S)"

# Create output directory
mkdir -p $OUTPUT_DIR

# Training with Simple TempMe (recommended)
python main.py \
    --do_train 1 \
    --datatype msrvtt \
    --train_csv $DATA_DIR/MSRVTT_train.9k.csv \
    --val_csv $DATA_DIR/MSRVTT_JSFUSION_test.csv \
    --data_path $DATA_DIR/MSRVTT_data.json \
    --features_path $DATA_DIR/videos \
    --use_tempme True \
    --tempme_type simple \
    --tempme_compression_ratio 0.75 \
    --tempme_hidden_dim 256 \
    --max_frames 12 \
    --max_words 32 \
    --batch_size 128 \
    --batch_size_val 1000 \
    --lr 1e-4 \
    --warmup_proportion 0.1 \
    --epochs 5 \
    --optim AdamW \
    --cross_model cross-base \
    --pretrained_clip_name ViT-B/32 \
    --output_dir $OUTPUT_DIR \
    --n_display 50

echo "Training completed. Results saved to: $OUTPUT_DIR"

# Optional: Run evaluation
echo "Running evaluation..."
python main.py \
    --do_eval 1 \
    --datatype msrvtt \
    --val_csv $DATA_DIR/MSRVTT_JSFUSION_test.csv \
    --data_path $DATA_DIR/MSRVTT_data.json \
    --features_path $DATA_DIR/videos \
    --use_tempme True \
    --tempme_type simple \
    --tempme_compression_ratio 0.75 \
    --max_frames 12 \
    --batch_size_val 1000 \
    --resume $OUTPUT_DIR/ckpt.pth.tar \
    --output_dir $OUTPUT_DIR/evaluation

echo "Evaluation completed."
