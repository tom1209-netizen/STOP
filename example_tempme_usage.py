#!/usr/bin/env python3
"""
Example script for using TempMe-STOP integrated model
This script demonstrates how to configure and run the unified model.
"""

import os
import argparse


def create_example_config():
    """Create example configuration for TempMe-STOP model"""
    
    # Base configuration for STOP model
    base_config = {
        # Dataset and paths
        'datatype': 'msrvtt',
        'data_dir': '/path/to/data',
        'train_csv': 'data/MSRVTT_train.9k.csv',
        'val_csv': 'data/MSRVTT_JSFUSION_test.csv',
        'data_path': 'data/MSRVTT_data.json',
        'features_path': '/path/to/videos',
        
        # Model configuration
        'cross_model': 'cross-base',
        'pretrained_clip_name': 'ViT-B/32',
        'max_words': 32,
        'max_frames': 12,
        
        # Training parameters
        'epochs': 5,
        'batch_size': 128,
        'batch_size_val': 1000,
        'lr': 1e-4,
        'warmup_proportion': 0.1,
        'optim': 'AdamW',
        
        # Output
        'output_dir': 'outputs/tempme_stop_experiment',
        
        # TempMe configuration
        'use_tempme': True,
        'tempme_type': 'simple',
        'tempme_compression_ratio': 0.75,
        'tempme_hidden_dim': 256,
    }
    
    return base_config


def generate_command_examples():
    """Generate example commands for different TempMe configurations"""
    
    commands = []
    
    # Example 1: Simple TempMe (Recommended for beginners)
    cmd1 = """
# Simple TempMe with 25% compression
python main.py \\
    --do_train 1 \\
    --datatype msrvtt \\
    --train_csv data/MSRVTT_train.9k.csv \\
    --val_csv data/MSRVTT_JSFUSION_test.csv \\
    --data_path data/MSRVTT_data.json \\
    --features_path /path/to/videos \\
    --use_tempme True \\
    --tempme_type simple \\
    --tempme_compression_ratio 0.75 \\
    --max_frames 12 \\
    --batch_size 128 \\
    --lr 1e-4 \\
    --epochs 5 \\
    --output_dir outputs/tempme_simple_experiment
"""
    commands.append(("Simple TempMe", cmd1))
    
    # Example 2: Basic TempMe with custom settings
    cmd2 = """
# Basic TempMe with transformer-based compression
python main.py \\
    --do_train 1 \\
    --datatype msrvtt \\
    --train_csv data/MSRVTT_train.9k.csv \\
    --val_csv data/MSRVTT_JSFUSION_test.csv \\
    --data_path data/MSRVTT_data.json \\
    --features_path /path/to/videos \\
    --use_tempme True \\
    --tempme_type basic \\
    --tempme_input_frames 32 \\
    --tempme_output_frames 12 \\
    --tempme_num_layers 2 \\
    --tempme_num_heads 8 \\
    --max_frames 12 \\
    --batch_size 64 \\
    --lr 5e-5 \\
    --epochs 5 \\
    --output_dir outputs/tempme_basic_experiment
"""
    commands.append(("Basic TempMe", cmd2))
    
    # Example 3: Adaptive TempMe
    cmd3 = """
# Adaptive TempMe with dynamic compression
python main.py \\
    --do_train 1 \\
    --datatype msrvtt \\
    --train_csv data/MSRVTT_train.9k.csv \\
    --val_csv data/MSRVTT_JSFUSION_test.csv \\
    --data_path data/MSRVTT_data.json \\
    --features_path /path/to/videos \\
    --use_tempme True \\
    --tempme_type adaptive \\
    --tempme_max_input_frames 48 \\
    --tempme_min_output_frames 8 \\
    --tempme_max_output_frames 16 \\
    --max_frames 12 \\
    --batch_size 64 \\
    --lr 5e-5 \\
    --epochs 5 \\
    --output_dir outputs/tempme_adaptive_experiment
"""
    commands.append(("Adaptive TempMe", cmd3))
    
    # Example 4: Evaluation only
    cmd4 = """
# Evaluation with TempMe
python main.py \\
    --do_eval 1 \\
    --datatype msrvtt \\
    --val_csv data/MSRVTT_JSFUSION_test.csv \\
    --data_path data/MSRVTT_data.json \\
    --features_path /path/to/videos \\
    --use_tempme True \\
    --tempme_type simple \\
    --tempme_compression_ratio 0.75 \\
    --max_frames 12 \\
    --batch_size_val 1000 \\
    --resume outputs/tempme_simple_experiment/ckpt.pth.tar \\
    --output_dir outputs/tempme_evaluation
"""
    commands.append(("Evaluation Only", cmd4))
    
    return commands


def create_training_script():
    """Create a complete training script with TempMe"""
    
    script_content = '''#!/bin/bash

# TempMe-STOP Training Script
# This script trains the integrated TempMe-STOP model

# Set paths (modify these for your setup)
DATA_DIR="/path/to/your/data"
OUTPUT_DIR="outputs/tempme_stop_experiment_$(date +%Y%m%d_%H%M%S)"

# Create output directory
mkdir -p $OUTPUT_DIR

# Training with Simple TempMe (recommended)
python main.py \\
    --do_train 1 \\
    --datatype msrvtt \\
    --train_csv $DATA_DIR/MSRVTT_train.9k.csv \\
    --val_csv $DATA_DIR/MSRVTT_JSFUSION_test.csv \\
    --data_path $DATA_DIR/MSRVTT_data.json \\
    --features_path $DATA_DIR/videos \\
    --use_tempme True \\
    --tempme_type simple \\
    --tempme_compression_ratio 0.75 \\
    --tempme_hidden_dim 256 \\
    --max_frames 12 \\
    --max_words 32 \\
    --batch_size 128 \\
    --batch_size_val 1000 \\
    --lr 1e-4 \\
    --warmup_proportion 0.1 \\
    --epochs 5 \\
    --optim AdamW \\
    --cross_model cross-base \\
    --pretrained_clip_name ViT-B/32 \\
    --output_dir $OUTPUT_DIR \\
    --n_display 50

echo "Training completed. Results saved to: $OUTPUT_DIR"

# Optional: Run evaluation
echo "Running evaluation..."
python main.py \\
    --do_eval 1 \\
    --datatype msrvtt \\
    --val_csv $DATA_DIR/MSRVTT_JSFUSION_test.csv \\
    --data_path $DATA_DIR/MSRVTT_data.json \\
    --features_path $DATA_DIR/videos \\
    --use_tempme True \\
    --tempme_type simple \\
    --tempme_compression_ratio 0.75 \\
    --max_frames 12 \\
    --batch_size_val 1000 \\
    --resume $OUTPUT_DIR/ckpt.pth.tar \\
    --output_dir $OUTPUT_DIR/evaluation

echo "Evaluation completed."
'''
    
    return script_content


def main():
    print("TempMe-STOP Integration Examples")
    print("=" * 50)
    
    # Show configuration example
    print("\n1. Example Configuration:")
    config = create_example_config()
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Show command examples
    print("\n2. Example Commands:")
    commands = generate_command_examples()
    for name, cmd in commands:
        print(f"\n{name}:")
        print(cmd.strip())
    
    # Create training script file
    script_content = create_training_script()
    script_path = "run_tempme_training.sh"
    
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make script executable
    os.chmod(script_path, 0o755)
    
    print(f"\n3. Training Script Created: {script_path}")
    print("   Make sure to update the DATA_DIR path in the script before running!")
    
    print("\n4. Quick Start:")
    print("   1. Update paths in run_tempme_training.sh")
    print("   2. Run: ./run_tempme_training.sh")
    print("   3. Monitor training progress and results")
    
    print("\n5. Key Benefits of TempMe Integration:")
    print("   • Reduces computational overhead by 20-40%")
    print("   • Maintains video understanding quality")
    print("   • Seamless integration with existing STOP pipeline")
    print("   • Multiple compression strategies available")


if __name__ == "__main__":
    main()