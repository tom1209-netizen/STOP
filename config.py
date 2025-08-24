"""
Configuration classes for the Unified TempMe-STOP Model
"""

from dataclasses import dataclass
from typing import Optional, List
import json
import os


@dataclass
class TempMeConfig:
    """Configuration for TempMe frame compression module."""
    
    # Model architecture
    base_encoder: str = "ViT-B/32"
    pretrained_path: str = "./pretrained"
    
    # LoRA configuration
    lora_dim: int = 8
    
    # Token merging configuration
    tome_r: int = 2
    merge_frame_num: str = "2-2-2"  # Frame merging at different layers
    merge_layer: List[int] = None  # Layers where frame merging occurs
    merge_token_proportion: List[float] = None  # Token merging proportions
    
    # Frame processing
    frame_list: List[int] = None
    patch_list: List[int] = None
    frame_pos: int = 1
    
    def __post_init__(self):
        """Set default values for list fields."""
        if self.merge_layer is None:
            self.merge_layer = [6, 8, 10]
        if self.merge_token_proportion is None:
            self.merge_token_proportion = [0.3, 0.7]
        if self.frame_list is None:
            self.frame_list = [32, 16, 8, 4]  # Frame count at different layers
        if self.patch_list is None:
            self.patch_list = [197, 98, 49, 25]  # Patch count at different layers


@dataclass  
class STOPConfig:
    """Configuration for STOP retrieval module."""
    
    # Model architecture
    cross_model: str = "cross-base"
    pretrained_clip_name: str = "ViT-B/32"
    pretrained_dir: str = "./pretrained"
    cache_dir: str = "./cache"
    
    # Training parameters
    freeze_clip: bool = False
    freeze_layer_num: int = 0
    
    # Architecture details
    sim_header: str = "meanP"  # Similarity header type
    pre_visual_pooling: bool = False
    
    # Prompting configuration
    temporal_prompt: str = "group2-2"
    new_added_modules: List[str] = None
    
    # Loss configuration
    loss_type: str = "CrossEn"
    
    def __post_init__(self):
        """Set default values for list fields."""
        if self.new_added_modules is None:
            self.new_added_modules = ["prompt", "temporal"]


@dataclass
class UnifiedModelConfig:
    """Configuration for the unified TempMe-STOP model."""
    
    # Sub-module configurations
    tempme: TempMeConfig = None
    stop: STOPConfig = None
    
    # Unified model parameters
    device: str = "cuda"
    max_frames: int = 12  # Maximum frames after compression
    
    # Video processing
    video_size: int = 224
    video_fps: float = 3.0
    num_segments: int = 32  # Initial sampling before compression
    
    # Inference parameters
    top_k: int = 5
    similarity_threshold: float = 0.5
    
    def __post_init__(self):
        """Initialize sub-configurations if not provided."""
        if self.tempme is None:
            self.tempme = TempMeConfig()
        if self.stop is None:
            self.stop = STOPConfig()
    
    @classmethod
    def from_json(cls, config_path: str) -> 'UnifiedModelConfig':
        """Load configuration from JSON file."""
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        # Parse TempMe config
        tempme_dict = config_dict.get('tempme', {})
        tempme_config = TempMeConfig(**tempme_dict)
        
        # Parse STOP config
        stop_dict = config_dict.get('stop', {})
        stop_config = STOPConfig(**stop_dict)
        
        # Parse unified config
        unified_dict = {k: v for k, v in config_dict.items() 
                       if k not in ['tempme', 'stop']}
        unified_dict['tempme'] = tempme_config
        unified_dict['stop'] = stop_config
        
        return cls(**unified_dict)
    
    def to_json(self, config_path: str):
        """Save configuration to JSON file."""
        # Convert to dictionary
        config_dict = {
            'tempme': self.tempme.__dict__,
            'stop': self.stop.__dict__,
            'device': self.device,
            'max_frames': self.max_frames,
            'video_size': self.video_size,
            'video_fps': self.video_fps,
            'num_segments': self.num_segments,
            'top_k': self.top_k,
            'similarity_threshold': self.similarity_threshold
        }
        
        # Save to file
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)


def get_default_config() -> UnifiedModelConfig:
    """Get default configuration for the unified model."""
    return UnifiedModelConfig()


def create_sample_config(output_path: str = "./configs/unified_model_config.json"):
    """Create a sample configuration file."""
    config = get_default_config()
    config.to_json(output_path)
    print(f"Sample configuration saved to: {output_path}")


if __name__ == "__main__":
    # Create sample configuration
    create_sample_config()