"""
Vision encoder with innate priors.

This encoder applies biologically-inspired innate processing BEFORE 
any learned features:

1. Color opponency (retinal processing)
2. Gabor filters (V1 edge detection)
3. Depth cues (monocular depth perception)
4. Spatial priors (3D understanding)

Then learned convolutional features extract higher-level representations.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.priors.visual_prior import ColorOpponencyPrior, GaborPrior, DepthCuesPrior
from src.priors.spatial_prior import SpatialPrior3D


@dataclass
class VisionEncoderConfig:
    """Configuration for vision encoder."""
    input_height: int = 224
    input_width: int = 224
    input_channels: int = 3
    latent_dim: int = 512
    num_gabor_orientations: int = 8
    num_gabor_scales: int = 4
    gabor_kernel_size: int = 15
    use_spatial_prior: bool = True
    use_color_prior: bool = True
    use_gabor_prior: bool = True
    use_depth_prior: bool = True


class ResBlock(nn.Module):
    """Residual block with optional downsampling."""
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        stride: int = 1
    ) -> None:
        super().__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.activation = nn.GELU()
        
        # Shortcut connection
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = out + identity
        out = self.activation(out)
        
        return out


class VisionEncoderWithPriors(nn.Module):
    """
    Vision encoder that applies innate priors BEFORE learned features.
    
    Pipeline:
    1. Color opponency (innate) -> 3 opponent channels [L, RG, BY]
    2. Gabor filters (innate) -> Edge/texture features
    3. Depth cues (innate) -> Monocular depth prior
    4. Concatenate all prior features
    5. Learned encoder -> Abstract features
    6. Spatial prior -> Position encoding
    
    This architecture ensures the encoder has innate structure
    before any learning occurs.
    """
    
    def __init__(self, config: VisionEncoderConfig) -> None:
        super().__init__()
        
        self.config = config
        
        # Calculate number of input channels to learned encoder
        # Start with opponent colors (3 channels)
        prior_channels = 3 if config.use_color_prior else config.input_channels
        
        # Add Gabor filter outputs
        if config.use_gabor_prior:
            num_gabor_filters = config.num_gabor_orientations * config.num_gabor_scales
            prior_channels += num_gabor_filters
        
        # Add depth prior (1 channel)
        if config.use_depth_prior:
            prior_channels += 1
        
        self.prior_channels = prior_channels
        
        # ===== INNATE PRIORS (not trained) =====
        
        if config.use_color_prior:
            self.color_prior = ColorOpponencyPrior()
        else:
            self.color_prior = None
        
        if config.use_gabor_prior:
            self.gabor_prior = GaborPrior(
                num_orientations=config.num_gabor_orientations,
                num_scales=config.num_gabor_scales,
                kernel_size=config.gabor_kernel_size,
            )
        else:
            self.gabor_prior = None
        
        if config.use_depth_prior:
            self.depth_prior = DepthCuesPrior(config.input_height, config.input_width)
        else:
            self.depth_prior = None
        
        # ===== LEARNED ENCODER =====
        
        # Stem: reduce channels and spatial size
        self.conv_stem = nn.Sequential(
            nn.Conv2d(prior_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        
        # ResNet-style blocks
        self.layer1 = self._make_layer(64, 128, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(128, 256, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(256, 512, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(512, config.latent_dim, num_blocks=2, stride=1)
        
        # ===== SPATIAL PRIOR (applied to learned features) =====
        
        if config.use_spatial_prior:
            # Calculate output spatial size
            # Input: H x W -> stem (H/4 x W/4) -> layer2 (H/8 x W/8) -> layer3 (H/16 x W/16)
            out_h = config.input_height // 16
            out_w = config.input_width // 16
            self.spatial_prior = SpatialPrior3D(
                dim=config.latent_dim,
                max_height=out_h,
                max_width=out_w,
            )
        else:
            self.spatial_prior = None
        
        # Output dimension info
        self.out_channels = config.latent_dim
    
    def _make_layer(
        self, 
        in_channels: int, 
        out_channels: int, 
        num_blocks: int, 
        stride: int
    ) -> nn.Sequential:
        """Create a layer of ResBlocks."""
        layers = []
        
        # First block may downsample
        layers.append(ResBlock(in_channels, out_channels, stride=stride))
        
        # Remaining blocks maintain resolution
        for _ in range(1, num_blocks):
            layers.append(ResBlock(out_channels, out_channels, stride=1))
        
        return nn.Sequential(*layers)
    
    def apply_priors(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        Apply all innate priors to RGB input.
        
        Args:
            rgb: Input RGB image [B, 3, H, W]
            
        Returns:
            Prior features [B, prior_channels, H, W]
        """
        features = []
        
        # Color opponency
        if self.color_prior is not None:
            opponent = self.color_prior(rgb)  # [B, 3, H, W]
            features.append(opponent)
            
            # Extract luminance for Gabor
            luminance = opponent[:, 0:1, :, :]  # [B, 1, H, W]
        else:
            features.append(rgb)
            # Convert to grayscale for Gabor
            luminance = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
        
        # Gabor edge detection
        if self.gabor_prior is not None:
            gabor_features = self.gabor_prior(luminance)  # [B, N_filters, H, W]
            features.append(gabor_features)
        
        # Depth cue
        if self.depth_prior is not None:
            depth_cue = self.depth_prior(rgb)  # [B, 1, H, W]
            features.append(depth_cue)
        
        # Concatenate all prior features
        return torch.cat(features, dim=1)
    
    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        Encode RGB image to latent representation.
        
        Args:
            rgb: Input RGB image [B, 3, H, W] (values normalized)
            
        Returns:
            Latent spatial map [B, latent_dim, H', W']
        """
        # Apply innate priors
        prior_features = self.apply_priors(rgb)  # [B, prior_channels, H, W]
        
        # Learned encoding
        x = self.conv_stem(prior_features)  # [B, 64, H/4, W/4]
        x = self.layer1(x)  # [B, 128, H/4, W/4]
        x = self.layer2(x)  # [B, 256, H/8, W/8]
        x = self.layer3(x)  # [B, 512, H/16, W/16]
        x = self.layer4(x)  # [B, latent_dim, H/16, W/16]
        
        # Apply spatial prior to final features
        if self.spatial_prior is not None:
            x = self.spatial_prior(x)
        
        return x
    
    def get_pooled_features(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        Get globally pooled features.
        
        Args:
            rgb: Input RGB image [B, 3, H, W]
            
        Returns:
            Pooled features [B, latent_dim]
        """
        spatial_features = self.forward(rgb)  # [B, latent_dim, H', W']
        return spatial_features.mean(dim=(2, 3))  # [B, latent_dim]
    
    def output_shape(self, input_height: int, input_width: int) -> Tuple[int, int, int]:
        """
        Calculate output shape for given input size.
        
        Returns:
            Tuple of (channels, height, width)
        """
        out_h = input_height // 16
        out_w = input_width // 16
        return (self.config.latent_dim, out_h, out_w)


class VisionEncoderLite(nn.Module):
    """
    Lightweight vision encoder for faster inference.
    
    Uses fewer parameters while still applying innate priors.
    """
    
    def __init__(self, config: VisionEncoderConfig) -> None:
        super().__init__()
        
        self.config = config
        
        # Simpler prior setup - just color and depth
        self.color_prior = ColorOpponencyPrior()
        self.depth_prior = DepthCuesPrior(config.input_height, config.input_width)
        
        # 4 input channels: 3 opponent + 1 depth
        prior_channels = 4
        
        # Simpler encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(prior_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            
            nn.Conv2d(128, config.latent_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(config.latent_dim),
            nn.GELU(),
        )
        
        self.out_channels = config.latent_dim
    
    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        """Encode RGB to latent."""
        # Apply priors
        opponent = self.color_prior(rgb)
        depth = self.depth_prior(rgb)
        prior_features = torch.cat([opponent, depth], dim=1)
        
        # Encode
        return self.encoder(prior_features)
