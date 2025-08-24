#!/usr/bin/env python3
"""
Training script for the Unified TempMe-STOP Model

This script provides comprehensive training capabilities for the unified model with multiple training strategies:
- Joint training (both modules together)
- Individual module training (TempMe only or STOP only)
- Sequential training (TempMe first, then STOP)

Usage:
    # Joint training (recommended)
    python train_unified_model.py --config configs/training_config.json --mode joint
    
    # Train only TempMe module
    python train_unified_model.py --config configs/training_config.json --mode tempme_only
    
    # Train only STOP module
    python train_unified_model.py --config configs/training_config.json --mode stop_only
    
    # Sequential training
    python train_unified_model.py --config configs/training_config.json --mode sequential
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import logging
import argparse
import json
from tqdm import tqdm
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from unified_model import UnifiedTempMeSTOPModel, create_unified_model
from config import UnifiedModelConfig, UnifiedTrainingConfig
from dataloaders.data_dataloaders import DATALOADER_DICT

# Import modules needed for training
try:
    from stop_modules.simple_tokenizer import SimpleTokenizer as ClipTokenizer
except ImportError:
    try:
        from modules.simple_tokenizer import SimpleTokenizer as ClipTokenizer
    except ImportError:
        ClipTokenizer = None
        
try:
    from utils.optimization import BertAdam
except ImportError:
    BertAdam = None
    
try:
    from utils.lr_scheduler import lr_scheduler
except ImportError:
    lr_scheduler = None
    
try:
    from utils.misc import set_random_seed, save_checkpoint
except ImportError:
    def set_random_seed(seed):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    save_checkpoint = None
    
try:
    from utils.metrics import compute_metrics
except ImportError:
    compute_metrics = None

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class UnifiedModelTrainer:
    """Trainer class for the unified TempMe-STOP model."""
    
    def __init__(self, config: UnifiedModelConfig):
        self.config = config
        self.training_config = config.training
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        
        # Initialize model
        self.model = UnifiedTempMeSTOPModel(config)
        self.model.to(self.device)
        
        # Initialize tokenizer
        try:
            self.tokenizer = ClipTokenizer()
        except:
            logger.warning("ClipTokenizer not available, using dummy tokenizer")
            self.tokenizer = None
        
        # Training components
        self.optimizer = None
        self.scheduler = None
        self.scaler = torch.cuda.amp.GradScaler() if config.device == "cuda" else None
        
        # Tracking
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = 0.0
        
        # Setup directories
        self.setup_directories()
        
        # Setup tensorboard
        self.writer = SummaryWriter(os.path.join(self.training_config.checkpoint_dir, "logs"))
        
    def setup_directories(self):
        """Create necessary directories for training."""
        os.makedirs(self.training_config.checkpoint_dir, exist_ok=True)
        os.makedirs(os.path.join(self.training_config.checkpoint_dir, "logs"), exist_ok=True)
        
    def setup_dataloaders(self):
        """Setup training and validation dataloaders."""
        if not self.tokenizer:
            logger.error("Tokenizer not available, cannot setup dataloaders")
            return None, None
            
        # Mock args for compatibility with existing dataloaders
        class MockArgs:
            def __init__(self, config):
                self.datatype = config.training.datatype
                self.batch_size = config.training.batch_size
                self.batch_size_val = config.training.batch_size
                self.data_dir = config.training.train_data_path
                self.val_csv = config.training.val_data_path
                self.num_thread_reader = 4
                self.max_words = 77
                self.max_frames = config.max_frames
                self.video_size = config.video_size
                self.feature_framerate = 1
                # Add other required attributes as needed
                
        mock_args = MockArgs(self.config)
        
        try:
            # Get appropriate dataloader
            if self.training_config.datatype in DATALOADER_DICT:
                train_dataloader, train_length, train_sampler = DATALOADER_DICT[self.training_config.datatype]["train"](
                    mock_args, self.tokenizer
                )
                val_dataloader, val_length = DATALOADER_DICT[self.training_config.datatype]["val"](
                    mock_args, self.tokenizer, subset="val"
                )
                
                logger.info(f"Loaded {train_length} training samples and {val_length} validation samples")
                return train_dataloader, val_dataloader
            else:
                logger.error(f"Datatype {self.training_config.datatype} not supported")
                return None, None
                
        except Exception as e:
            logger.error(f"Error setting up dataloaders: {e}")
            return None, None
    
    def setup_optimizer_and_scheduler(self):
        """Setup optimizer and learning rate scheduler."""
        training_mode = self.training_config.training_mode
        
        # Setup model parameters based on training mode
        if training_mode == "joint":
            self.model.unfreeze_all()
            param_groups = self.model.get_parameter_groups(
                self.training_config.tempme_lr,
                self.training_config.stop_lr
            )
        elif training_mode == "tempme_only":
            self.model.freeze_stop()
            param_groups = [{'params': self.model.get_trainable_parameters(), 'lr': self.training_config.tempme_lr}]
        elif training_mode == "stop_only":
            self.model.freeze_tempme()
            param_groups = [{'params': self.model.get_trainable_parameters(), 'lr': self.training_config.stop_lr}]
        else:
            raise ValueError(f"Unknown training mode: {training_mode}")
        
        # Setup optimizer
        if self.training_config.optimizer == "AdamW":
            self.optimizer = optim.AdamW(
                param_groups,
                weight_decay=self.training_config.weight_decay
            )
        elif self.training_config.optimizer == "BertAdam":
            # Flatten parameter groups for BertAdam
            all_params = []
            for group in param_groups:
                all_params.extend(group['params'])
            
            self.optimizer = BertAdam(
                all_params,
                lr=self.training_config.stop_lr,  # Use STOP lr as default
                warmup=self.training_config.warmup_epochs / self.training_config.num_epochs,
                schedule='warmup_cosine',
                weight_decay=self.training_config.weight_decay,
                max_grad_norm=self.training_config.max_grad_norm
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.training_config.optimizer}")
        
        # Setup scheduler
        if self.training_config.scheduler == "cosine" and self.training_config.optimizer == "AdamW":
            total_steps = self.training_config.num_epochs  # Simplified for now
            warmup_steps = int(self.training_config.warmup_epochs)
            
            self.scheduler = lr_scheduler(
                mode='cos',
                init_lr=self.training_config.stop_lr,
                all_iters=total_steps,
                slow_start_iters=warmup_steps,
                weight_decay=self.training_config.weight_decay
            )
        
        logger.info(f"Setup optimizer: {self.training_config.optimizer}")
        logger.info(f"Training mode: {training_mode}")
        
        # Log trainable parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Total params: {total_params:,}, Trainable: {trainable_params:,} "
                   f"({trainable_params/total_params*100:.2f}%)")
    
    def train_epoch(self, train_dataloader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_dataloader)
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {self.current_epoch+1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            if torch.cuda.is_available():
                batch = tuple(t.to(self.device) for t in batch)
            
            input_ids, input_mask, segment_ids, video, video_mask = batch
            
            # Forward pass
            self.optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                # Compute training loss
                loss_dict = self.model.compute_training_loss(
                    video, video_mask, input_ids, input_mask, segment_ids
                )
                
                total_loss_batch = loss_dict['total_loss']
                
                # Scale loss for gradient accumulation
                if self.training_config.gradient_accumulation_steps > 1:
                    total_loss_batch = total_loss_batch / self.training_config.gradient_accumulation_steps
            
            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(total_loss_batch).backward()
                
                if (batch_idx + 1) % self.training_config.gradient_accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.training_config.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    
                    if self.scheduler:
                        self.scheduler(self.optimizer, global_step=self.global_step)
                    
                    self.global_step += 1
            else:
                total_loss_batch.backward()
                
                if (batch_idx + 1) % self.training_config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.training_config.max_grad_norm)
                    self.optimizer.step()
                    
                    if self.scheduler:
                        self.scheduler(self.optimizer, global_step=self.global_step)
                    
                    self.global_step += 1
            
            # Update progress
            total_loss += total_loss_batch.item()
            progress_bar.set_postfix({
                'loss': f"{total_loss_batch.item():.4f}",
                'avg_loss': f"{total_loss/(batch_idx+1):.4f}"
            })
            
            # Log to tensorboard
            if self.global_step % 10 == 0:
                self.writer.add_scalar('Train/Loss', total_loss_batch.item(), self.global_step)
                for key, value in loss_dict.items():
                    if key != 'total_loss' and torch.is_tensor(value):
                        self.writer.add_scalar(f'Train/{key}', value.item(), self.global_step)
        
        avg_loss = total_loss / num_batches
        logger.info(f"Epoch {self.current_epoch+1} - Average training loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def evaluate(self, val_dataloader):
        """Evaluate the model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = len(val_dataloader)
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Evaluating"):
                if torch.cuda.is_available():
                    batch = tuple(t.to(self.device) for t in batch)
                
                input_ids, input_mask, segment_ids, video, video_mask = batch
                
                # Compute validation loss
                loss_dict = self.model.compute_training_loss(
                    video, video_mask, input_ids, input_mask, segment_ids
                )
                
                total_loss += loss_dict['total_loss'].item()
        
        avg_loss = total_loss / num_batches
        logger.info(f"Validation loss: {avg_loss:.4f}")
        
        # Log to tensorboard
        self.writer.add_scalar('Val/Loss', avg_loss, self.current_epoch)
        
        return avg_loss
    
    def save_checkpoint(self, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': self.best_metric,
            'config': self.config.__dict__
        }
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.training_config.checkpoint_dir, f"checkpoint_epoch_{self.current_epoch}.pth")
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.training_config.checkpoint_dir, "best_model.pth")
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model checkpoint: {best_path}")
        
        logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        if not os.path.exists(checkpoint_path):
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return False
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_metric = checkpoint['best_metric']
        
        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
        return True
    
    def train(self, resume_from=None):
        """Main training loop."""
        logger.info("Starting unified model training...")
        
        # Setup training components
        train_dataloader, val_dataloader = self.setup_dataloaders()
        if train_dataloader is None:
            logger.error("Failed to setup dataloaders")
            return
            
        self.setup_optimizer_and_scheduler()
        
        # Resume from checkpoint if specified
        if resume_from:
            self.load_checkpoint(resume_from)
            start_epoch = self.current_epoch + 1
        else:
            start_epoch = 0
        
        # Training loop
        for epoch in range(start_epoch, self.training_config.num_epochs):
            self.current_epoch = epoch
            
            # Train for one epoch
            train_loss = self.train_epoch(train_dataloader)
            
            # Evaluate
            if epoch % self.training_config.eval_every_n_epochs == 0:
                val_loss = self.evaluate(val_dataloader)
                
                # Check if best model (using validation loss as metric)
                current_metric = -val_loss  # Negative because lower loss is better
                is_best = current_metric > self.best_metric
                if is_best:
                    self.best_metric = current_metric
                
                # Save checkpoint
                if epoch % self.training_config.save_every_n_epochs == 0:
                    self.save_checkpoint(is_best=is_best)
        
        # Final save
        self.save_checkpoint(is_best=False)
        self.writer.close()
        
        logger.info("Training completed!")


def main():
    parser = argparse.ArgumentParser(description="Train Unified TempMe-STOP Model")
    parser.add_argument("--config", type=str, required=True, 
                       help="Path to training configuration file")
    parser.add_argument("--mode", type=str, default="joint",
                       choices=["joint", "tempme_only", "stop_only", "sequential"],
                       help="Training mode")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seed
    set_random_seed(args.seed)
    
    # Load configuration
    if os.path.exists(args.config):
        config = UnifiedModelConfig.from_json(args.config)
    else:
        logger.error(f"Configuration file not found: {args.config}")
        return
    
    # Override training mode if specified
    config.training.training_mode = args.mode
    
    # Create trainer and start training
    trainer = UnifiedModelTrainer(config)
    
    if args.mode == "sequential":
        # Sequential training: TempMe first, then STOP
        logger.info("Starting sequential training...")
        
        # Phase 1: Train TempMe only
        logger.info("Phase 1: Training TempMe module...")
        config.training.training_mode = "tempme_only"
        config.training.num_epochs = config.training.num_epochs // 2
        trainer_tempme = UnifiedModelTrainer(config)
        trainer_tempme.train(args.resume)
        
        # Phase 2: Train STOP only (load TempMe weights)
        logger.info("Phase 2: Training STOP module...")
        tempme_checkpoint = os.path.join(config.training.checkpoint_dir, "best_model.pth")
        config.training.training_mode = "stop_only"
        trainer_stop = UnifiedModelTrainer(config)
        trainer_stop.train(tempme_checkpoint)
    else:
        # Standard training
        trainer.train(args.resume)


if __name__ == "__main__":
    main()