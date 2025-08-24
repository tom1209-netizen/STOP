#!/usr/bin/env python3
"""
Simple training example for the Unified TempMe-STOP Model

This script demonstrates how to quickly start training the unified model
with minimal configuration.
"""

import os
import sys
import torch
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from unified_model import create_unified_model
from config import UnifiedModelConfig, UnifiedTrainingConfig
from train_unified_model import UnifiedModelTrainer


def quick_train_example():
    """
    Quick training example with default settings.
    Demonstrates the simplest way to start training.
    """
    print("🚀 Quick Training Example for Unified TempMe-STOP Model")
    print("=" * 60)
    
    # Create default configuration
    config = UnifiedModelConfig()
    
    # Customize training settings
    config.training.training_mode = "joint"
    config.training.batch_size = 4  # Small batch for demo
    config.training.num_epochs = 5  # Short training for demo
    config.training.tempme_lr = 1e-4
    config.training.stop_lr = 1e-5
    config.training.checkpoint_dir = "./demo_checkpoints"
    
    # Set data paths (adjust these to your actual data)
    config.training.datatype = "msrvtt"  # Change to your dataset
    config.training.train_data_path = "./data/train"
    config.training.val_data_path = "./data/val"
    
    print(f"Training Configuration:")
    print(f"  - Mode: {config.training.training_mode}")
    print(f"  - Epochs: {config.training.num_epochs}")
    print(f"  - Batch Size: {config.training.batch_size}")
    print(f"  - TempMe LR: {config.training.tempme_lr}")
    print(f"  - STOP LR: {config.training.stop_lr}")
    print(f"  - Dataset: {config.training.datatype}")
    print()
    
    # Create trainer
    try:
        trainer = UnifiedModelTrainer(config)
        print("✅ Model and trainer initialized successfully")
        
        # Start training
        print("🏋️  Starting training...")
        trainer.train()
        
        print("🎉 Training completed!")
        print(f"📁 Checkpoints saved to: {config.training.checkpoint_dir}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        print("💡 Tips:")
        print("  - Make sure your data paths are correct")
        print("  - Check that you have enough GPU memory")
        print("  - Try reducing batch_size if you get OOM errors")


def train_with_custom_config():
    """
    Training example using a custom configuration file.
    """
    print("⚙️  Training with Custom Configuration")
    print("=" * 50)
    
    config_path = "./configs/training_config.json"
    
    if not os.path.exists(config_path):
        print(f"❌ Configuration file not found: {config_path}")
        print("💡 Creating a sample configuration...")
        
        # Create sample config
        config = UnifiedModelConfig()
        os.makedirs("./configs", exist_ok=True)
        config.to_json(config_path)
        print(f"✅ Sample configuration created: {config_path}")
        print("📝 Please edit the configuration file and run again")
        return
    
    # Load configuration
    config = UnifiedModelConfig.from_json(config_path)
    print(f"✅ Loaded configuration from: {config_path}")
    
    # Create trainer and start training
    trainer = UnifiedModelTrainer(config)
    trainer.train()


def demonstrate_training_modes():
    """
    Demonstrate different training modes.
    """
    print("🎯 Training Mode Demonstration")
    print("=" * 40)
    
    modes = ["joint", "tempme_only", "stop_only"]
    
    for mode in modes:
        print(f"\n🔄 Training Mode: {mode}")
        print("-" * 30)
        
        # Create configuration for this mode
        config = UnifiedModelConfig()
        config.training.training_mode = mode
        config.training.num_epochs = 2  # Very short for demo
        config.training.batch_size = 2  # Small batch
        config.training.checkpoint_dir = f"./demo_checkpoints/{mode}"
        
        print(f"Configuration:")
        print(f"  - Mode: {config.training.training_mode}")
        print(f"  - Epochs: {config.training.num_epochs}")
        print(f"  - Checkpoint Dir: {config.training.checkpoint_dir}")
        
        try:
            # Create model to check parameter counts
            model = create_unified_model()
            
            if mode == "joint":
                model.unfreeze_all()
            elif mode == "tempme_only":
                model.freeze_stop()
            elif mode == "stop_only":
                model.freeze_tempme()
            
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            print(f"  - Total Params: {total_params:,}")
            print(f"  - Trainable: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
            
            # Note: In a real demo, you would run trainer.train() here
            print(f"✅ {mode} mode configured successfully")
            
        except Exception as e:
            print(f"❌ Error in {mode} mode: {e}")


def main():
    """
    Main function - choose your training approach.
    """
    print("🎓 Unified TempMe-STOP Model Training Examples")
    print("=" * 60)
    print()
    print("Choose a training example:")
    print("1. Quick training with default settings")
    print("2. Training with custom configuration file")
    print("3. Demonstrate different training modes")
    print("4. View training guide")
    print()
    
    while True:
        choice = input("Enter your choice (1-4): ").strip()
        
        if choice == "1":
            quick_train_example()
            break
        elif choice == "2":
            train_with_custom_config()
            break
        elif choice == "3":
            demonstrate_training_modes()
            break
        elif choice == "4":
            print("\n📖 Training Guide:")
            print("Please see TRAINING_GUIDE.md for comprehensive documentation")
            if os.path.exists("TRAINING_GUIDE.md"):
                print("✅ TRAINING_GUIDE.md found in current directory")
            else:
                print("❌ TRAINING_GUIDE.md not found")
            break
        else:
            print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")


if __name__ == "__main__":
    main()