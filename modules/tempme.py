"""
TempMe: Temporal Memory Module for Video Frame Compression
This module performs learnable temporal compression to reduce the number of frames
while preserving important temporal information for the STOP model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TempMeCompressor(nn.Module):
    """
    TempMe module for temporal frame compression.
    Takes dense video frames and outputs compressed representative frames.
    """
    
    def __init__(self, 
                 input_frames=32,      # Number of input dense frames
                 output_frames=12,     # Number of output compressed frames  
                 frame_dim=768,        # Frame feature dimension
                 hidden_dim=512,       # Hidden dimension for compression
                 num_layers=3,         # Number of transformer layers
                 num_heads=8):         # Number of attention heads
        super(TempMeCompressor, self).__init__()
        
        self.input_frames = input_frames
        self.output_frames = output_frames
        self.frame_dim = frame_dim
        self.hidden_dim = hidden_dim
        
        # Temporal encoding for input frames
        self.temporal_embedding = nn.Parameter(torch.randn(input_frames, frame_dim))
        
        # Learnable query tokens for compressed frames
        self.compression_queries = nn.Parameter(torch.randn(output_frames, frame_dim))
        
        # Transformer layers for temporal attention and compression
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=frame_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=0.1,
            batch_first=True
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Cross-attention for frame compression
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=frame_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=0.1,
            batch_first=True
        )
        self.compression_decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
        
        # Frame importance scoring
        self.importance_scorer = nn.Sequential(
            nn.Linear(frame_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize model parameters"""
        nn.init.normal_(self.temporal_embedding, std=0.02)
        nn.init.normal_(self.compression_queries, std=0.02)
    
    def forward(self, dense_frames):
        """
        Forward pass for TempMe compression
        
        Args:
            dense_frames: Tensor of shape [B, T_in, C, H, W] where T_in = input_frames
        
        Returns:
            compressed_frames: Tensor of shape [B, T_out, C, H, W] where T_out = output_frames
            compression_weights: Attention weights for interpretability
        """
        B, T_in, C, H, W = dense_frames.shape
        assert T_in == self.input_frames, f"Expected {self.input_frames} input frames, got {T_in}"
        
        # Flatten spatial dimensions for processing
        frames_flat = dense_frames.view(B, T_in, -1)  # [B, T_in, C*H*W]
        
        # Project to frame_dim if necessary
        if frames_flat.size(-1) != self.frame_dim:
            if not hasattr(self, 'frame_projector'):
                self.frame_projector = nn.Linear(frames_flat.size(-1), self.frame_dim).to(frames_flat.device)
            frames_flat = self.frame_projector(frames_flat)
        
        # Add temporal embeddings
        frames_encoded = frames_flat + self.temporal_embedding.unsqueeze(0)  # [B, T_in, frame_dim]
        
        # Apply temporal encoding to understand frame relationships
        temporal_features = self.temporal_encoder(frames_encoded)  # [B, T_in, frame_dim]
        
        # Compute frame importance scores
        importance_scores = self.importance_scorer(temporal_features)  # [B, T_in, 1]
        
        # Apply importance weighting
        weighted_features = temporal_features * importance_scores
        
        # Prepare compression queries
        queries = self.compression_queries.unsqueeze(0).expand(B, -1, -1)  # [B, T_out, frame_dim]
        
        # Cross-attention based compression
        compressed_features = self.compression_decoder(
            tgt=queries,
            memory=weighted_features
        )  # [B, T_out, frame_dim]
        
        # Project back to original frame space and reshape
        if hasattr(self, 'frame_projector'):
            if not hasattr(self, 'frame_deprojector'):
                self.frame_deprojector = nn.Linear(self.frame_dim, C*H*W).to(compressed_features.device)
            compressed_flat = self.frame_deprojector(compressed_features)
        else:
            compressed_flat = compressed_features
        
        # Reshape back to frame format
        compressed_frames = compressed_flat.view(B, self.output_frames, C, H, W)
        
        # Return compression weights for interpretability
        compression_weights = importance_scores.squeeze(-1)  # [B, T_in]
        
        return compressed_frames, compression_weights


class AdaptiveTempMe(nn.Module):
    """
    Adaptive TempMe that can handle variable input lengths and 
    dynamically adjust compression ratios.
    """
    
    def __init__(self, 
                 max_input_frames=64,
                 min_output_frames=8,
                 max_output_frames=16,
                 frame_channels=3,
                 frame_size=224):
        super(AdaptiveTempMe, self).__init__()
        
        self.max_input_frames = max_input_frames
        self.min_output_frames = min_output_frames
        self.max_output_frames = max_output_frames
        self.frame_channels = frame_channels
        self.frame_size = frame_size
        
        # Spatial feature extractor
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(frame_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128), 
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d((7, 7))
        )
        
        spatial_feature_dim = 256 * 7 * 7
        
        # Temporal compression network
        self.temporal_compressor = nn.Sequential(
            nn.Linear(spatial_feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256)
        )
        
        # Attention-based frame selection
        self.frame_selector = nn.MultiheadAttention(
            embed_dim=256,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Frame reconstruction
        self.frame_reconstructor = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, spatial_feature_dim)
        )
        
        # Spatial decoder
        self.spatial_decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(32, frame_channels, kernel_size=7, stride=2, padding=3, output_padding=1),
            nn.Tanh()
        )
    
    def forward(self, dense_frames, target_frames=None):
        """
        Forward pass for adaptive TempMe
        
        Args:
            dense_frames: [B, T_in, C, H, W]
            target_frames: Target number of output frames (optional)
        
        Returns:
            compressed_frames: [B, T_out, C, H, W]
        """
        B, T_in, C, H, W = dense_frames.shape
        
        # Determine target number of frames
        if target_frames is None:
            # Adaptive compression ratio based on input length
            compression_ratio = max(0.3, min(0.8, self.min_output_frames / T_in))
            target_frames = max(self.min_output_frames, 
                               min(self.max_output_frames, int(T_in * compression_ratio)))
        
        # Extract spatial features
        frames_reshaped = dense_frames.view(B * T_in, C, H, W)
        spatial_features = self.spatial_encoder(frames_reshaped)  # [B*T_in, 256, 7, 7]
        spatial_features = spatial_features.view(B, T_in, -1)  # [B, T_in, 256*7*7]
        
        # Temporal compression
        temporal_features = self.temporal_compressor(spatial_features)  # [B, T_in, 256]
        
        # Frame selection using attention
        # Use learnable queries for target number of frames
        queries = torch.randn(B, target_frames, 256, device=dense_frames.device)
        selected_features, attention_weights = self.frame_selector(
            query=queries,
            key=temporal_features,
            value=temporal_features
        )  # [B, target_frames, 256]
        
        # Reconstruct frames
        reconstructed_features = self.frame_reconstructor(selected_features)  # [B, target_frames, 256*7*7]
        reconstructed_features = reconstructed_features.view(B * target_frames, 256, 7, 7)
        
        # Decode to frame space
        compressed_frames = self.spatial_decoder(reconstructed_features)  # [B*target_frames, C, H, W]
        compressed_frames = compressed_frames.view(B, target_frames, C, H, W)
        
        return compressed_frames


class SimpleTempMe(nn.Module):
    """
    Simplified TempMe for integration with existing STOP pipeline.
    Works with existing frame counts and performs temporal compression.
    """
    
    def __init__(self, compression_ratio=0.75, hidden_dim=256):
        super(SimpleTempMe, self).__init__()
        self.compression_ratio = compression_ratio
        self.hidden_dim = hidden_dim
        
        # Spatial feature extractor (lightweight)
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((7, 7))
        )
        
        # Temporal attention for frame importance
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=128 * 7 * 7,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Frame importance scorer
        self.importance_scorer = nn.Sequential(
            nn.Linear(128 * 7 * 7, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Spatial decoder
        self.spatial_decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )
        
    def forward(self, video_frames):
        """
        Args:
            video_frames: [B, T, C, H, W]
        Returns:
            compressed_frames: [B, T_compressed, C, H, W]
        """
        B, T, C, H, W = video_frames.shape
        
        # Calculate target frames based on compression ratio
        target_frames = max(1, int(T * self.compression_ratio))
        
        if target_frames >= T:
            # No compression needed
            return video_frames
        
        # Extract spatial features
        frames_flat = video_frames.view(B * T, C, H, W)
        spatial_features = self.spatial_encoder(frames_flat)  # [B*T, 128, 7, 7]
        feature_dim = spatial_features.size(1) * spatial_features.size(2) * spatial_features.size(3)
        spatial_features = spatial_features.view(B, T, feature_dim)  # [B, T, 128*7*7]
        
        # Compute frame importance
        importance_scores = self.importance_scorer(spatial_features)  # [B, T, 1]
        importance_scores = importance_scores.squeeze(-1)  # [B, T]
        
        # Select most important frames
        _, top_indices = torch.topk(importance_scores, target_frames, dim=1, sorted=True)
        
        # Sort indices to maintain temporal order
        top_indices_sorted, _ = torch.sort(top_indices, dim=1)
        
        # Select frames based on importance
        batch_indices = torch.arange(B, device=video_frames.device).unsqueeze(1).expand(-1, target_frames)
        selected_frames = video_frames[batch_indices, top_indices_sorted]  # [B, target_frames, C, H, W]
        
        return selected_frames


def get_tempme_model(config):
    """Factory function to create TempMe model based on configuration"""
    model_type = getattr(config, 'tempme_type', 'simple')
    
    if model_type == 'simple':
        return SimpleTempMe(
            compression_ratio=getattr(config, 'tempme_compression_ratio', 0.75),
            hidden_dim=getattr(config, 'tempme_hidden_dim', 256)
        )
    elif model_type == 'basic':
        return TempMeCompressor(
            input_frames=getattr(config, 'tempme_input_frames', 32),
            output_frames=getattr(config, 'tempme_output_frames', 12),
            frame_dim=getattr(config, 'tempme_frame_dim', 768),
            hidden_dim=getattr(config, 'tempme_hidden_dim', 512),
            num_layers=getattr(config, 'tempme_num_layers', 3),
            num_heads=getattr(config, 'tempme_num_heads', 8)
        )
    elif model_type == 'adaptive':
        return AdaptiveTempMe(
            max_input_frames=getattr(config, 'tempme_max_input_frames', 64),
            min_output_frames=getattr(config, 'tempme_min_output_frames', 8),
            max_output_frames=getattr(config, 'tempme_max_output_frames', 16),
            frame_channels=getattr(config, 'tempme_frame_channels', 3),
            frame_size=getattr(config, 'tempme_frame_size', 224)
        )
    else:
        raise ValueError(f"Unknown TempMe model type: {model_type}")