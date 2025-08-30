# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

from typing import Tuple
import torch
import torch.nn.functional as F
from .clip import ResidualAttentionBlock
from .tome_merge import bipartite_soft_matching, merge_source, merge_wavg
import logging

logger = logging.getLogger(__name__)

class ToMeResidualAttentionBlock(ResidualAttentionBlock):
    """
    ToMe-enhanced ResidualAttentionBlock for STOP model.
    Supports both intra-frame and inter-frame token merging.
    """

    def forward(self, x_tuple: tuple):
        x, video_frame, visual = x_tuple
        
        if not visual:
            # For text processing, use original forward
            return super().forward(x_tuple)
        
        # Get layer-specific ToMe configuration
        M_frame_num = getattr(self, '_M_frame_num', 1)
        M_token_num = getattr(self, '_M_token_num', [])
        frame_pos = self._tome_info.get("frame_pos", 0) if hasattr(self, '_tome_info') else 0
        
        # Inter-frame merging (when M_frame_num > 1)
        if M_frame_num > 1 and hasattr(self, '_tome_info') and len(M_token_num) > 0:
            r_f = M_frame_num
            r = M_token_num[0]
            
            # Get dimensions - x is [seq_len, batch_size, embed_dim]
            seq_len, batch_size, embed_dim = x.shape
            cls_num = getattr(self, 'visual_prompt_length', 1)
            
            if self._tome_info.get("size") is None:
                self._tome_info["size"] = torch.ones(batch_size, seq_len, 1, device=x.device)
            
            # Reshape for inter-frame processing
            # batch_size should be divisible by r_f for frame merging
            if batch_size % r_f == 0:
                # Transpose to [batch, seq, embed] for reshaping
                x = x.permute(1, 0, 2)
                metric = x.detach()
                
                # Reshape to group frames together
                x = x.reshape(batch_size // r_f, r_f, seq_len, embed_dim)
                metric = metric.reshape(batch_size // r_f, r_f, seq_len, embed_dim)
                info_size = self._tome_info["size"].reshape(batch_size // r_f, r_f, seq_len, 1)
                
                # Separate CLS tokens and patch tokens across frames
                x_cls = x[:, :, :cls_num, :].reshape(batch_size // r_f, -1, embed_dim)
                x_patch = x[:, :, cls_num:, :].reshape(batch_size // r_f, -1, embed_dim)
                x = torch.cat([x_cls, x_patch], dim=1)
                
                metric_cls = metric[:, :, :cls_num, :].reshape(batch_size // r_f, -1, embed_dim)
                metric_patch = metric[:, :, cls_num:, :].reshape(batch_size // r_f, -1, embed_dim)
                metric = torch.cat([metric_cls, metric_patch], dim=1)
                
                info_size_cls = info_size[:, :, :cls_num, :].reshape(batch_size // r_f, -1, 1)
                info_size_patch = info_size[:, :, cls_num:, :].reshape(batch_size // r_f, -1, 1)
                self._tome_info["size"] = torch.cat([info_size_cls, info_size_patch], dim=1)
                
                # Update tome info for inter-frame merging
                self._tome_info["cls_num"] = cls_num * r_f
                self._tome_info["token_num"] = seq_len * r_f
                self._tome_info["frame_num"] = self._tome_info.get("frame_num", video_frame) // r_f
                
                if r > 0:
                    # Apply inter-frame token merging
                    merge, _ = bipartite_soft_matching(
                        metric,
                        r,
                        cls_num=self._tome_info["cls_num"],
                        r_f=r_f
                    )
                    
                    if self._tome_info.get("trace_source", False):
                        self._tome_info["source"] = merge_source(
                            merge, x, self._tome_info.get("source")
                        )
                    
                    x, self._tome_info["size"] = merge_wavg(merge, x, self._tome_info["size"])
                    self._tome_info["token_num"] = self._tome_info["token_num"] - r
                
                # Transpose back to [seq, batch, embed]
                x = x.permute(1, 0, 2)
                batch_size = batch_size // r_f  # Update batch size after merging
        
        # Standard STOP attention mechanism
        if visual and hasattr(self, 'visual_prompt_length'):
            # Use STOP's visual attention with prompts
            B = x.size(1)
            BT = B * video_frame
            T = video_frame
            dim = x.size(-1)
            
            visual_prompt, frame_token = x[:self.visual_prompt_length, :, :], x[self.visual_prompt_length:, :, :].reshape(-1, BT, dim)
            frame_token_ln = self.ln_1(frame_token)
            visual_prompt_ln = self.ln_1(visual_prompt)
            
            query1 = frame_token_ln
            key1 = torch.zeros(self.visual_prompt_length + query1.size(0), BT, dim).to(x.device)
            for i in range(0, BT, B):
                key1[:, i:i+B, :] = torch.cat((visual_prompt_ln, query1[:, i:i+B, :]), dim=0)
            
            attention_output_frames = self.attention(query1, key1, key1).reshape(-1, B, dim)
            query2 = visual_prompt_ln
            key2 = torch.cat((visual_prompt_ln, frame_token_ln.reshape(-1, B, dim)), dim=0).to(x.device)
            attention_output_prompt = self.attention(query2, key2, key2)
            
            x = x + torch.cat((attention_output_prompt, attention_output_frames), dim=0)
            
            # Get metric for potential post-attention merging
            metric = torch.cat((attention_output_prompt, attention_output_frames), dim=0)
        else:
            # Standard attention
            x_ln = self.ln_1(x)
            attn_output = self.attention(x_ln, x_ln, x_ln)
            x = x + attn_output
            metric = attn_output

        # Intra-frame token merging (after attention)
        if len(M_token_num) > 1 and hasattr(self, '_tome_info'):
            r = M_token_num[-1]
            if r > 0:
                # Apply intra-frame token merging
                x_transposed = x.permute(1, 0, 2)  # [batch, seq, embed]
                metric_transposed = metric.detach().permute(1, 0, 2)
                
                current_cls_num = self._tome_info.get("cls_num", getattr(self, 'visual_prompt_length', 1))
                
                merge, _ = bipartite_soft_matching(
                    metric_transposed,
                    r,
                    cls_num=current_cls_num,
                    r_f=1
                )
                
                if self._tome_info.get("trace_source", False):
                    self._tome_info["source"] = merge_source(
                        merge, x_transposed, self._tome_info.get("source")
                    )
                
                x_merged, self._tome_info["size"] = merge_wavg(
                    merge, x_transposed, self._tome_info.get("size")
                )
                x = x_merged.permute(1, 0, 2)  # Back to [seq, batch, embed]
                self._tome_info["token_num"] = self._tome_info.get("token_num", x.size(0)) - r
        elif len(M_token_num) == 1 and hasattr(self, '_tome_info'):
            # Simple intra-frame merging for non-inter-frame layers
            r = M_token_num[0]
            if r > 0:
                x_transposed = x.permute(1, 0, 2)  # [batch, seq, embed]
                metric_transposed = metric.detach().permute(1, 0, 2)
                
                current_cls_num = getattr(self, 'visual_prompt_length', 1)
                
                merge, _ = bipartite_soft_matching(
                    metric_transposed,
                    r,
                    cls_num=current_cls_num,
                    r_f=1
                )
                
                if self._tome_info.get("trace_source", False):
                    self._tome_info["source"] = merge_source(
                        merge, x_transposed, self._tome_info.get("source")
                    )
                
                x_merged, self._tome_info["size"] = merge_wavg(
                    merge, x_transposed, self._tome_info.get("size")
                )
                x = x_merged.permute(1, 0, 2)  # Back to [seq, batch, embed]

        # MLP
        if hasattr(self, 'LoRA') and getattr(self.args, 'lora', False):
            x = x + self.mlp(self.ln_2(x)) + 0.1 * self.LoRA(x)
        else:
            x = x + self.mlp(self.ln_2(x))
        
        return (x, video_frame, visual)


def apply_tome_patch(model, trace_source: bool = False, prop_attn: bool = True, 
                     tome_r: int = 2, merge_layers: list = None, merge_frame_nums: list = None,
                     merge_token_proportions: list = None, frame_pos: int = 0):
    """
    Apply ToMe patch to STOP's CLIP model with support for both intra and inter-frame merging.
    
    Args:
        model: The CLIP model from STOP
        trace_source: Whether to trace source tokens
        prop_attn: Whether to propagate attention with size information
        tome_r: Number of tokens to merge per layer (for layers without inter-frame merging)
        merge_layers: List of layer indices where inter-frame merging occurs
        merge_frame_nums: List of frame merge ratios for each merge layer
        merge_token_proportions: List of token merge proportions [inter_frame_prop, intra_frame_prop]
        frame_pos: Whether to use positional embeddings (0=no, 1=yes)
    """
    # Set default configurations
    if merge_layers is None:
        merge_layers = [3, 6, 9]  # Default merge at layers 3, 6, 9
    if merge_frame_nums is None:
        merge_frame_nums = [2, 2, 2]  # Default merge 2 frames at each layer
    if merge_token_proportions is None:
        merge_token_proportions = [0.1, 0.1]  # Default 10% token merging for both inter and intra
    
    # Initialize tome info on the model
    model._tome_info = {
        "frame_num": 0,
        "token_num": 0,
        "cls_num": 1,  # Typically 1 CLS token for CLIP
        "size": None,
        "source": None,
        "trace_source": trace_source,
        "prop_attn": prop_attn,
        "tome_r": tome_r,
        "merge_layers": merge_layers,
        "merge_frame_nums": merge_frame_nums.copy(),  # Make a copy since we'll pop from it
        "merge_token_proportions": merge_token_proportions,
        "frame_pos": frame_pos
    }

    # Patch visual transformer blocks
    if hasattr(model, 'visual') and hasattr(model.visual, 'transformer'):
        transformer_blocks = model.visual.transformer.resblocks
    elif hasattr(model, 'clip') and hasattr(model.clip, 'visual'):
        transformer_blocks = model.clip.visual.transformer.resblocks
    else:
        logger.warning("Could not find transformer blocks to patch")
        return

    # Patch each transformer block with layer-specific ToMe configuration
    for layer_idx, module in enumerate(transformer_blocks):
        if isinstance(module, ResidualAttentionBlock):
            # Replace the class to ToMeResidualAttentionBlock
            module.__class__ = ToMeResidualAttentionBlock
            module._tome_info = model._tome_info
            module._layer_idx = layer_idx
            
            # Configure layer-specific merging parameters
            if layer_idx in merge_layers:
                merge_idx = merge_layers.index(layer_idx)
                module._M_frame_num = merge_frame_nums[merge_idx] if merge_idx < len(merge_frame_nums) else 1
                # Inter-frame and intra-frame token merging
                inter_ratio = int(197 * module._M_frame_num * merge_token_proportions[0])  # Assuming ViT-B patch count
                intra_ratio = int(197 * merge_token_proportions[1])
                module._M_token_num = [inter_ratio, intra_ratio]
            else:
                module._M_frame_num = 1
                # Only intra-frame merging
                if layer_idx < merge_layers[0] if merge_layers else False:
                    module._M_token_num = [tome_r]  # Use basic tome_r
                else:
                    # Post inter-frame merging layers
                    intra_ratio = int(197 * merge_token_proportions[1])
                    module._M_token_num = [intra_ratio]

    logger.info(f"Applied ToMe patch with r={tome_r}, merge_layers={merge_layers}, "
                f"merge_frame_nums={merge_frame_nums}, trace_source={trace_source}, prop_attn={prop_attn}")


def calculate_merge_schedule(num_layers: int, max_frames: int, initial_tokens: int, 
                            merge_layers: list, merge_frame_nums: list, 
                            merge_token_proportions: list, tome_r: int):
    """
    Calculate the token and frame reduction schedule across transformer layers.
    
    Returns:
        patch_list: Number of patches/tokens at each layer
        frame_list: Number of frames at each layer  
    """
    patch_list = [initial_tokens]
    frame_list = [max_frames]
    
    patch_num = initial_tokens
    frame_num = max_frames
    merge_frame_nums_copy = merge_frame_nums.copy()
    
    for layer in range(num_layers):
        if layer not in merge_layers:
            if layer < merge_layers[0] if merge_layers else False:
                # Pre inter-frame merging: only basic token merging
                patch_num = patch_num - tome_r
                patch_list.append(patch_num)
                frame_list.append(frame_num)
            else:
                # Post inter-frame merging: intra-frame token merging
                patch_num = patch_num - int(patch_num * merge_token_proportions[1])
                patch_list.append(patch_num)
                frame_list.append(frame_num)
        else:
            # Inter-frame merging layer
            M_frame_num = merge_frame_nums_copy.pop(0)
            M_token_num = int(patch_num * M_frame_num * merge_token_proportions[0])
            
            assert frame_num % M_frame_num == 0, f"Frame number {frame_num} not divisible by merge ratio {M_frame_num}"
            patch_num = patch_num * M_frame_num - M_token_num
            frame_num = frame_num // M_frame_num
            patch_list.append(patch_num)
            frame_list.append(frame_num)
            
            # Additional intra-frame merging after inter-frame merging
            patch_num = patch_num - int(patch_num * merge_token_proportions[1])
            patch_list.append(patch_num)
            frame_list.append(frame_num)
    
    return patch_list, frame_list
