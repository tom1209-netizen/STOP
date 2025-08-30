# Example configuration for STOP with complete ToMe integration
# Supports both intra-frame and inter-frame token merging

TOME_CONFIG = {
    # Basic ToMe Parameters
    "tome_r": 2,                    # Number of tokens to merge per layer (for non-inter-frame layers)
    "tome_trace_source": False,     # Whether to trace source tokens (for debugging)
    "tome_prop_attn": True,         # Whether to propagate attention with size information
    
    # Inter-frame Merging Parameters
    "merge_layers": [3, 6, 9],      # Layers where inter-frame merging occurs
    "merge_frame_nums": [2, 2, 2],  # Frame merge ratios for each merge layer
    "merge_token_proportions": [0.1, 0.1],  # [inter_frame_ratio, intra_frame_ratio]
    "frame_pos": 0,                 # Whether to use frame positional embeddings (0=no, 1=yes)
    
    # Existing STOP parameters that work with ToMe
    "max_frames": 12,               # Maximum number of video frames
    "pretrained_clip_name": "ViT-B/32",  # Base CLIP model
    "linear_patch": "2d",           # Patch processing method
    "sim_header": "meanP",          # Similarity header type
    
    # Training parameters
    "batch_size": 32,
    "learning_rate": 1e-4,
}

# Alternative configurations for different use cases:

# Memory-optimized configuration (aggressive merging)
MEMORY_OPTIMIZED_CONFIG = {
    "tome_r": 4,
    "merge_layers": [2, 4, 6, 8, 10],
    "merge_frame_nums": [2, 2, 3, 3, 3],
    "merge_token_proportions": [0.15, 0.15],
    "frame_pos": 0,
}

# Speed-optimized configuration (moderate merging)
SPEED_OPTIMIZED_CONFIG = {
    "tome_r": 2,
    "merge_layers": [3, 6, 9],
    "merge_frame_nums": [2, 2, 2],
    "merge_token_proportions": [0.08, 0.08],
    "frame_pos": 0,
}

# Accuracy-preserving configuration (conservative merging)
ACCURACY_PRESERVING_CONFIG = {
    "tome_r": 1,
    "merge_layers": [6, 9],
    "merge_frame_nums": [2, 2],
    "merge_token_proportions": [0.05, 0.05],
    "frame_pos": 1,  # Use positional embeddings for better accuracy
}

# Usage examples:

# Command Line Usage:
"""
# Basic inter-frame + intra-frame merging:
python main.py --tome_r 2 --merge_layers "3-6-9" --merge_frame_nums "2-2-2" --merge_token_proportions "10-10"

# Memory-optimized:
python main.py --tome_r 4 --merge_layers "2-4-6-8-10" --merge_frame_nums "2-2-3-3-3" --merge_token_proportions "15-15"

# Speed-optimized:
python main.py --tome_r 2 --merge_layers "3-6-9" --merge_frame_nums "2-2-2" --merge_token_proportions "8-8"

# Accuracy-preserving:
python main.py --tome_r 1 --merge_layers "6-9" --merge_frame_nums "2-2" --merge_token_proportions "5-5" --frame_pos 1
"""

# Programmatic Usage:
"""
# In your training script:
task_config.tome_r = 2
task_config.merge_layers = [3, 6, 9]
task_config.merge_frame_nums = [2, 2, 2]
task_config.merge_token_proportions = [0.1, 0.1]
task_config.frame_pos = 0

model = CLIP4Clip(cross_config, clip_state_dict, task_config)
"""
