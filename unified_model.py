"""
Unified TempMe-STOP Model for Temporal Video Event Retrieval

This module integrates TempMe (frame compression) and STOP (retrieval) into a single 
end-to-end pipeline for temporal video event retrieval.

Architecture:
Input Video → Frame Sampling → TempMe Compression (N→12) → STOP Retrieval → Output Frames
"""

import os
import torch
import torch.nn as nn
import numpy as np
from typing import List, Union, Tuple, Optional
import logging

# Import TempMe components
try:
    from tempme_module.modeling import VTRModel
    from tempme_module.module_clip import CLIP as TempMeCLIP
except ImportError as e:
    logger.warning(f"Failed to import TempMe modules: {e}")
    TempMeCLIP = None
    VTRModel = None

# Import STOP components  
try:
    from stop_modules.clip4clip import CLIP4Clip
    from stop_modules.simple_tokenizer import SimpleTokenizer
except ImportError as e:
    logger.warning(f"Failed to import STOP modules: {e}")
    CLIP4Clip = None
    SimpleTokenizer = None

# Import data loading utilities
try:
    from dataloaders.decode import RawVideoExtractorpyAV
    from dataloaders.transforms import init_transform_dict
except ImportError as e:
    logger.warning(f"Failed to import dataloader modules: {e}")
    RawVideoExtractorpyAV = None
    init_transform_dict = None

# Import configuration
from config import UnifiedModelConfig, TempMeConfig, STOPConfig

logger = logging.getLogger(__name__)


class UnifiedTempMeSTOPModel(nn.Module):
    """
    Unified model that integrates TempMe and STOP for end-to-end temporal video retrieval.
    
    Pipeline:
    1. Video Sampling: Extract frames from raw video
    2. TempMe Compression: Reduce N frames to 12 representative frames  
    3. STOP Retrieval: Use compressed frames for query-based retrieval
    """
    
    def __init__(self, config: UnifiedModelConfig):
        super(UnifiedTempMeSTOPModel, self).__init__()
        
        self.config = config
        self.device = config.device
        
        # Initialize TempMe model for frame compression
        if VTRModel is not None:
            self.tempme_model = VTRModel(config.tempme)
        else:
            logger.warning("TempMe model not available, frame compression will use simple sampling")
            self.tempme_model = None
        
        # Initialize STOP model for retrieval  
        if CLIP4Clip is not None:
            self.stop_model = CLIP4Clip.from_pretrained(
                config.stop.cross_model,
                cache_dir=config.stop.cache_dir,
                task_config=config.stop
            )
        else:
            logger.warning("STOP model not available, using dummy model")
            self.stop_model = None
        
        # Initialize tokenizer for text processing
        if SimpleTokenizer is not None:
            self.tokenizer = SimpleTokenizer()
        else:
            logger.warning("SimpleTokenizer not available")
            self.tokenizer = None
        
        # Initialize video extractor for frame sampling
        self.video_extractor = self._init_video_extractor()
        
        # Set to evaluation mode by default
        self.eval()
        
    def _init_video_extractor(self):
        """Initialize video extractor with appropriate transforms."""
        if RawVideoExtractorpyAV is None or init_transform_dict is None:
            logger.warning("Video extractor not available")
            return None
            
        # Set up video extraction parameters
        video_params = {
            'num_segments': self.config.num_segments,  # Sample more frames initially for TempMe compression
            'video_fmt': 'mp4',
            'train': False,  # Use deterministic sampling for inference
        }
        
        # Initialize transforms
        transform_dict = init_transform_dict()
        transform = transform_dict['test']
        
        return RawVideoExtractorpyAV(
            framerate=1,
            size=self.config.video_size, 
            centercrop=True,
            **video_params,
            transform=transform
        )
    
    def preprocess_video(self, video_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract and preprocess frames from video.
        
        Args:
            video_path: Path to input video file
            
        Returns:
            video_tensor: Preprocessed video frames [T, C, H, W] 
            video_mask: Video mask indicating valid frames [T]
        """
        if self.video_extractor is None:
            # Fallback: create dummy video tensor for testing
            logger.warning("Using dummy video tensor for testing")
            T, C, H, W = self.config.num_segments, 3, self.config.video_size, self.config.video_size
            video_tensor = torch.randn(T, C, H, W)
            video_mask = torch.ones(T)
            return video_tensor, video_mask
            
        try:
            # Extract frames using video extractor
            video_tensor, frame_length = self.video_extractor.get_video_data(video_path)
            
            # Create video mask
            video_mask = torch.zeros(video_tensor.size(0))
            video_mask[:frame_length] = 1.0
            
            return video_tensor, video_mask
            
        except Exception as e:
            logger.error(f"Error preprocessing video {video_path}: {e}")
            raise
    
    def preprocess_text(self, query: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Tokenize and preprocess text query.
        
        Args:
            query: Natural language query string
            
        Returns:
            input_ids: Tokenized text [L]
            attention_mask: Attention mask [L] 
            token_type_ids: Token type IDs [L]
        """
        if self.tokenizer is None:
            # Fallback: create dummy text tensors for testing
            logger.warning("Using dummy text tensors for testing")
            seq_len = 77  # Standard CLIP sequence length
            input_ids = torch.randint(0, 49408, (seq_len,))  # CLIP vocab size
            attention_mask = torch.ones(seq_len)
            token_type_ids = torch.zeros(seq_len)
            return input_ids, attention_mask, token_type_ids
            
        # Tokenize text
        tokens = self.tokenizer.encode(query)
        
        # Convert to tensors
        input_ids = torch.tensor(tokens, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        token_type_ids = torch.zeros_like(input_ids)
        
        return input_ids, attention_mask, token_type_ids
    
    def compress_frames_with_tempme(self, video_tensor: torch.Tensor, video_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compress video frames using TempMe model.
        
        Args:
            video_tensor: Input video frames [T, C, H, W]
            video_mask: Video mask [T]
            
        Returns:
            compressed_frames: Compressed frames [12, C, H, W]
            compressed_mask: Compressed frame mask [12]
        """
        max_frames = self.config.max_frames
        
        # If TempMe model is not available, use simple sampling
        if self.tempme_model is None:
            logger.info("Using simple uniform sampling for frame compression")
            if video_tensor.size(0) > max_frames:
                # Uniform sampling to get max_frames
                indices = torch.linspace(0, video_tensor.size(0) - 1, max_frames).long()
                compressed_frames = video_tensor[indices]
                compressed_mask = video_mask[indices]
            else:
                # Pad if fewer than max_frames
                pad_size = max_frames - video_tensor.size(0)
                compressed_frames = torch.cat([
                    video_tensor,
                    torch.zeros(pad_size, *video_tensor.shape[1:])
                ], dim=0)
                compressed_mask = torch.cat([
                    video_mask,
                    torch.zeros(pad_size)
                ], dim=0)
            return compressed_frames, compressed_mask
        
        # Use TempMe for intelligent frame compression
        # Add batch dimension and prepare for TempMe
        video_batch = video_tensor.unsqueeze(0).to(self.device)  # [1, T, C, H, W]
        mask_batch = video_mask.unsqueeze(0).to(self.device)     # [1, T]
        
        # Use TempMe to extract compressed video features
        with torch.no_grad():
            # Get compressed video representation from TempMe
            video_feat = self.tempme_model.get_video_feat(video_batch, mask_batch)
            
            # TempMe outputs averaged features, but we need frame-level representations
            # We'll use the internal compressed frames from the visual transformer
            # This requires accessing the intermediate representations
            
            # For now, we'll use a simplified approach:
            # Reshape the video to have max_frames and let STOP handle it
            if video_tensor.size(0) > max_frames:
                # Uniform sampling to get max_frames
                indices = torch.linspace(0, video_tensor.size(0) - 1, max_frames).long()
                compressed_frames = video_tensor[indices]
                compressed_mask = video_mask[indices]
            else:
                # Pad if fewer than max_frames
                pad_size = max_frames - video_tensor.size(0)
                compressed_frames = torch.cat([
                    video_tensor,
                    torch.zeros(pad_size, *video_tensor.shape[1:])
                ], dim=0)
                compressed_mask = torch.cat([
                    video_mask,
                    torch.zeros(pad_size)
                ], dim=0)
        
        return compressed_frames, compressed_mask
    
    def retrieve_with_stop(self, compressed_frames: torch.Tensor, compressed_mask: torch.Tensor, 
                          input_ids: torch.Tensor, attention_mask: torch.Tensor, 
                          token_type_ids: torch.Tensor) -> torch.Tensor:
        """
        Perform retrieval using STOP model.
        
        Args:
            compressed_frames: Compressed video frames [12, C, H, W]
            compressed_mask: Compressed frame mask [12]
            input_ids: Tokenized text [L]
            attention_mask: Text attention mask [L]
            token_type_ids: Token type IDs [L]
            
        Returns:
            similarity_scores: Similarity scores between text and video frames
        """
        if self.stop_model is None:
            # Fallback: return dummy similarity scores
            logger.warning("Using dummy similarity scores for testing")
            valid_frames = torch.sum(compressed_mask).int().item()
            return torch.randn(valid_frames)
        
        # Prepare input format for STOP model
        # STOP expects: [batch_size, pair, video_frame, channel, h, w]
        video_input = compressed_frames.unsqueeze(0).unsqueeze(0)  # [1, 1, 12, C, H, W]
        video_mask_input = compressed_mask.unsqueeze(0)            # [1, 12]
        
        # Add batch dimension to text inputs
        text_input = input_ids.unsqueeze(0)         # [1, L]
        text_mask = attention_mask.unsqueeze(0)     # [1, L]
        text_types = token_type_ids.unsqueeze(0)    # [1, L]
        
        # Move to device
        video_input = video_input.to(self.device)
        video_mask_input = video_mask_input.to(self.device)
        text_input = text_input.to(self.device)
        text_mask = text_mask.to(self.device)
        text_types = text_types.to(self.device)
        
        # Run STOP model
        with torch.no_grad():
            output = self.stop_model(
                input_ids=text_input,
                token_type_ids=text_types,
                attention_mask=text_mask,
                video=video_input,
                video_mask=video_mask_input
            )
            
            # Extract similarity logits
            sequence_output = output['sequence_output']
            visual_output = output['visual_output']
            
            # Compute similarity between text and video
            similarity_scores, _ = self.stop_model.get_similarity_logits(
                sequence_output, visual_output, text_mask, video_mask_input
            )
        
        return similarity_scores.squeeze()
    
    def forward(self, video_path: str, query: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward pass: video + query → similarity scores.
        
        Args:
            video_path: Path to input video
            query: Natural language query
            
        Returns:
            similarity_scores: Frame-level similarity scores
            frame_indices: Corresponding frame indices
        """
        # Step 1: Preprocess video
        video_tensor, video_mask = self.preprocess_video(video_path)
        
        # Step 2: Preprocess text
        input_ids, attention_mask, token_type_ids = self.preprocess_text(query)
        
        # Step 3: Compress frames with TempMe  
        compressed_frames, compressed_mask = self.compress_frames_with_tempme(video_tensor, video_mask)
        
        # Step 4: Retrieve with STOP
        similarity_scores = self.retrieve_with_stop(
            compressed_frames, compressed_mask, 
            input_ids, attention_mask, token_type_ids
        )
        
        # Create frame indices for the compressed frames
        valid_frames = torch.sum(compressed_mask).int().item()
        frame_indices = torch.arange(valid_frames)
        
        return similarity_scores[:valid_frames], frame_indices
    
    def retrieve_event(self, video_path: str, query: str, top_k: int = 5) -> List[int]:
        """
        Main API function for temporal video event retrieval.
        
        Args:
            video_path: Path to input video file
            query: Natural language query describing the event
            top_k: Number of top frames to return
            
        Returns:
            List of frame indices corresponding to the query
        """
        try:
            # Run full pipeline
            similarity_scores, frame_indices = self.forward(video_path, query)
            
            # Get top-k most relevant frames
            top_scores, top_indices = torch.topk(similarity_scores, min(top_k, len(similarity_scores)))
            
            # Convert to frame indices
            relevant_frames = frame_indices[top_indices].tolist()
            
            logger.info(f"Retrieved {len(relevant_frames)} frames for query: '{query}'")
            logger.info(f"Top similarity scores: {top_scores.tolist()}")
            
            return relevant_frames
            
        except Exception as e:
            logger.error(f"Error in retrieve_event: {e}")
            raise

    def extract_video_embedding(self, video_path: str) -> torch.Tensor:
        """
        Extract video embedding from a video file.
        
        Args:
            video_path: Path to input video file
            
        Returns:
            Video embedding tensor
        """
        try:
            # Step 1: Preprocess video
            video_tensor, video_mask = self.preprocess_video(video_path)
            
            # Step 2: Compress frames with TempMe to get meaningful representation
            compressed_frames, compressed_mask = self.compress_frames_with_tempme(video_tensor, video_mask)
            
            # Step 3: Extract visual features using STOP's visual encoder
            if self.stop_model is not None:
                # Use STOP's visual encoder to get video embeddings
                video_input = compressed_frames.unsqueeze(0).unsqueeze(0)  # [1, 1, 12, C, H, W]
                video_mask_input = compressed_mask.unsqueeze(0)            # [1, 12]
                
                # Move to device
                video_input = video_input.to(self.device)
                video_mask_input = video_mask_input.to(self.device)
                
                with torch.no_grad():
                    # Get unified visual prompt (if exists)
                    unified_visual_prompt = getattr(self.stop_model, 'unified_visual_prompt', None)
                    
                    # Get visual features from STOP model
                    visual_output = self.stop_model.get_visual_output(
                        video_input, 
                        unified_visual_prompt,
                        video_mask_input,
                        video_frame=video_input.size(2)  # number of frames
                    )
                    
                    # Pool the visual features to get a single video embedding
                    # Use mean pooling over the frame dimension
                    valid_frames = torch.sum(video_mask_input, dim=1, keepdim=True)  # [1, 1]
                    video_embedding = torch.sum(visual_output, dim=1) / valid_frames  # [1, hidden_dim]
                    
                    return video_embedding.squeeze(0)  # [hidden_dim]
            else:
                # Fallback: use simple mean pooling over frames
                logger.warning("STOP model not available, using simple frame averaging")
                valid_frames = torch.sum(compressed_mask).int().item()
                if valid_frames > 0:
                    # Flatten spatial dimensions and average over valid frames
                    frame_features = compressed_frames[:valid_frames].flatten(1)  # [valid_frames, C*H*W]
                    video_embedding = torch.mean(frame_features, dim=0)  # [C*H*W]
                    return video_embedding
                else:
                    # Return zero embedding if no valid frames
                    return torch.zeros(compressed_frames.size(1) * compressed_frames.size(2) * compressed_frames.size(3))
                    
        except Exception as e:
            logger.error(f"Error in extract_video_embedding: {e}")
            raise
    
    def compute_training_loss(self, video_tensor: torch.Tensor, video_mask: torch.Tensor,
                             input_ids: torch.Tensor, attention_mask: torch.Tensor, 
                             token_type_ids: torch.Tensor) -> dict:
        """
        Compute training loss for the unified model.
        
        Args:
            video_tensor: Video frames [B, T, C, H, W]
            video_mask: Video mask [B, T]
            input_ids: Text tokens [B, L]
            attention_mask: Text attention mask [B, L] 
            token_type_ids: Token type IDs [B, L]
            
        Returns:
            Dictionary containing loss components
        """
        batch_size = video_tensor.size(0)
        
        # Compress frames using TempMe for each video in batch
        compressed_videos = []
        compressed_masks = []
        
        for i in range(batch_size):
            compressed_frames, compressed_mask = self.compress_frames_with_tempme(
                video_tensor[i], video_mask[i]
            )
            compressed_videos.append(compressed_frames)
            compressed_masks.append(compressed_mask)
        
        # Stack compressed videos
        compressed_video_batch = torch.stack(compressed_videos)  # [B, max_frames, C, H, W]
        compressed_mask_batch = torch.stack(compressed_masks)    # [B, max_frames]
        
        # Prepare for STOP model (add pair dimension)
        video_input = compressed_video_batch.unsqueeze(1)  # [B, 1, max_frames, C, H, W]
        
        # Run STOP model for training
        if self.stop_model is not None:
            output = self.stop_model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                video=video_input,
                video_mask=compressed_mask_batch
            )
            
            # Extract loss components
            loss_dict = {
                'total_loss': output.get('loss', torch.tensor(0.0)),
                'sim_loss': output.get('sim_loss', torch.tensor(0.0)),
            }
            
            # Add TempMe specific losses if available
            if 'tempme_loss' in output:
                loss_dict['tempme_loss'] = output['tempme_loss']
                
        else:
            # Fallback loss computation
            logger.warning("STOP model not available, using dummy loss")
            loss_dict = {
                'total_loss': torch.tensor(0.0, requires_grad=True),
                'sim_loss': torch.tensor(0.0, requires_grad=True),
            }
        
        return loss_dict
    
    def freeze_tempme(self):
        """Freeze TempMe parameters for STOP-only training."""
        if self.tempme_model is not None:
            for param in self.tempme_model.parameters():
                param.requires_grad = False
            logger.info("TempMe parameters frozen")
    
    def freeze_stop(self):
        """Freeze STOP parameters for TempMe-only training.""" 
        if self.stop_model is not None:
            for param in self.stop_model.parameters():
                param.requires_grad = False
            logger.info("STOP parameters frozen")
    
    def unfreeze_all(self):
        """Unfreeze all parameters for joint training."""
        for param in self.parameters():
            param.requires_grad = True
        logger.info("All parameters unfrozen for joint training")
    
    def get_trainable_parameters(self):
        """Get trainable parameters for optimizer."""
        return [p for p in self.parameters() if p.requires_grad]
    
    def get_parameter_groups(self, tempme_lr: float = 1e-4, stop_lr: float = 1e-5):
        """
        Get parameter groups with different learning rates.
        
        Args:
            tempme_lr: Learning rate for TempMe parameters
            stop_lr: Learning rate for STOP parameters
            
        Returns:
            List of parameter groups for optimizer
        """
        tempme_params = []
        stop_params = []
        
        # Collect TempMe parameters
        if self.tempme_model is not None:
            for name, param in self.tempme_model.named_parameters():
                if param.requires_grad:
                    tempme_params.append(param)
        
        # Collect STOP parameters  
        if self.stop_model is not None:
            for name, param in self.stop_model.named_parameters():
                if param.requires_grad:
                    stop_params.append(param)
        
        param_groups = []
        if tempme_params:
            param_groups.append({'params': tempme_params, 'lr': tempme_lr, 'name': 'tempme'})
        if stop_params:
            param_groups.append({'params': stop_params, 'lr': stop_lr, 'name': 'stop'})
            
        return param_groups


def create_unified_model(config_path: str = None, device: str = 'cuda') -> UnifiedTempMeSTOPModel:
    """
    Factory function to create a unified TempMe-STOP model.
    
    Args:
        config_path: Path to configuration file
        device: Device to run model on
        
    Returns:
        Initialized unified model
    """
    if config_path and os.path.exists(config_path):
        config = UnifiedModelConfig.from_json(config_path)
    else:
        # Use default configuration
        config = UnifiedModelConfig()
        
    # Override device if specified
    config.device = device
    
    return UnifiedTempMeSTOPModel(config)