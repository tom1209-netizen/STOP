# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

def parse_r(num_layers: int, r: int):
    """
    Process a constant value of r or a list of r values.
    
    Args:
        num_layers: Number of layers in the model
        r: Either an integer or a list of integers for each layer
        
    Returns:
        A list of r values for each layer
    """
    if isinstance(r, list):
        if len(r) < num_layers:
            r = r + [0] * (num_layers - len(r))
        return r[:num_layers]
    else:
        return [r] * num_layers

def init_tome_info(model, max_frames: int = 12, initial_tokens: int = 197):
    """
    Initialize ToMe information for the model.
    
    Args:
        model: The model to initialize
        max_frames: Maximum number of frames
        initial_tokens: Initial number of tokens (e.g., 197 for ViT-B/16)
    """
    if not hasattr(model, '_tome_info'):
        model._tome_info = {}
    
    model._tome_info.update({
        "frame_num": max_frames,
        "token_num": initial_tokens,
        "cls_num": 1,
        "size": None,
        "source": None
    })
