#!/bin/bash

# Example training script for STOP with ToMe integration
# This script demonstrates how to train the STOP model with Token Merging enabled

set -e

# Dataset and output configuration
DATA_PATH="data/MSRVTT_data.json"
FEATURES_PATH="data/MSRVTT_Videos"
OUTPUT_DIR="output/stop_tome_experiment"

# Create output directory
mkdir -p $OUTPUT_DIR

# Training with ToMe enabled (both intra and inter-frame merging)
echo "Training STOP with ToMe (r=2, inter-frame merging)..."
python main.py \
    --do_train 1 \
    --data_path $DATA_PATH \
    --features_path $FEATURES_PATH \
    --output_dir $OUTPUT_DIR \
    --max_words 32 \
    --max_frames 12 \
    --batch_size 128 \
    --batch_size_val 1000 \
    --lr 1e-4 \
    --epochs 5 \
    --n_display 100 \
    --tome_r 2 \
    --tome_prop_attn \
    --merge_layers "3-6-9" \
    --merge_frame_nums "2-2-2" \
    --merge_token_proportions "10-10" \
    --frame_pos 0 \
    --pretrained_clip_name "ViT-B/32" \
    --sim_header "meanP" \
    --temporal_prompt "DGL" \
    --linear_patch "2d" \
    --loose_type \
    --datatype "msrvtt" \
    --task_type "retrieval" \
    --world_size 1 \
    --local_rank 0

echo "Training completed! Model saved to $OUTPUT_DIR"

# Optional: Run evaluation with different ToMe settings
echo "Running evaluation with different ToMe settings..."

# Test different configurations
configs=(
    "0 1-1-1 5-5"        # No ToMe
    "2 1-1-1 5-5"        # Basic intra-frame only  
    "2 2-2-2 10-10"      # Inter-frame + intra-frame
    "4 2-3-3 15-15"      # Aggressive merging
)

for config in "${configs[@]}"; do
    IFS=' ' read -r tome_r merge_nums merge_props <<< "$config"
    echo "Evaluating with tome_r=$tome_r, merge_frame_nums=$merge_nums, token_proportions=$merge_props"
    
    python main.py \
        --do_eval 1 \
        --data_path $DATA_PATH \
        --features_path $FEATURES_PATH \
        --output_dir "${OUTPUT_DIR}_eval_r${tome_r}_${merge_nums// /_}" \
        --init_model "$OUTPUT_DIR/pytorch_model.bin.1" \
        --max_words 32 \
        --max_frames 12 \
        --batch_size_val 1000 \
        --tome_r $tome_r \
        --tome_prop_attn \
        --merge_layers "3-6-9" \
        --merge_frame_nums "$merge_nums" \
        --merge_token_proportions "$merge_props" \
        --frame_pos 0 \
        --pretrained_clip_name "ViT-B/32" \
        --sim_header "meanP" \
        --temporal_prompt "DGL" \
        --linear_patch "2d" \
        --loose_type \
        --datatype "msrvtt" \
        --task_type "retrieval" \
        --world_size 1 \
        --local_rank 0
done

echo "All evaluations completed!"
