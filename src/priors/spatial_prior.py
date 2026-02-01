"""
Spatial innate priors - understanding 3D structure from 2D images.

These encode fundamental spatial reasoning that appears to be largely innate:

1. SpatialPrior3D: Encodes 3D spatial relationships
   - 2D rotary position embeddings (translation equivariance)
   - Perspective projection understanding
   - Occlusion reasoning (closer objects block farther)

2. RotaryEmbedding2D: Position encoding via rotation
   - Same object at different positions = same identity, just moved
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Import directly from module to avoid circular imports
from src.world_model.geometry import RotaryEmbedding2D
from src.world_model.config import GeometryConfig


class SpatialPrior3D(nn.Module):
    """
    Encode 3D spatial relationships as innate priors.
    
    Combines:
    - 2D Rotary embeddings (position encoding via rotation)
    - Perspective projection understanding (objects shrink with distance)
    - Scale consistency (same object at different depths = different sizes)
    
    This encodes the fundamental truth that visual space has structure,
    and position is not something to learn but something that IS.
    """
    
    def __init__(
        self, 
        dim: int, 
        max_height: int, 
        max_width: int,
        apply_perspective: bool = True,
    ) -> None:
        """
        Initialize 3D spatial prior.
        
        Args:
            dim: Channel dimension for rotary embeddings (must be even)
            max_height: Maximum spatial height
            max_width: Maximum spatial width
            apply_perspective: Whether to apply perspective scale modulation
        """
        super().__init__()
        
        self.dim = dim
        self.max_height = max_height
        self.max_width = max_width
        self.apply_perspective = apply_perspective
        
        # 2D Rotary position embeddings
        geometry_config = GeometryConfig(
            dim=dim,
            max_height=max_height,
            max_width=max_width,
        )
        self.rotary_2d = RotaryEmbedding2D(geometry_config)
        
        # Perspective prior: objects at top of image (far) appear smaller
        # This encodes the ground plane assumption
        if apply_perspective:
            scale_prior = self._create_perspective_prior(max_height, max_width)
            self.register_buffer('scale_prior', scale_prior)
        
        # Depth ordering prior: provides expected depth at each position
        depth_prior = self._create_depth_prior(max_height, max_width)
        self.register_buffer('depth_prior', depth_prior)
        
    def _create_perspective_prior(self, height: int, width: int) -> torch.Tensor:
        """
        Create perspective scale prior.
        
        Objects at the top of the image (assumed to be far away on a ground plane)
        should have their features scaled down to reflect foreshortening.
        
        Returns:
            Scale prior [1, 1, H, W]
        """
        # Linear scale from 1.0 (bottom, close) to 0.5 (top, far)
        y_scale = torch.linspace(0.5, 1.0, height).unsqueeze(1).expand(height, width)
        return y_scale.unsqueeze(0).unsqueeze(0)
    
    def _create_depth_prior(self, height: int, width: int) -> torch.Tensor:
        """
        Create depth ordering prior.
        
        Encodes expected relative depth at each spatial position.
        Lower in image = closer (higher depth value).
        
        Returns:
            Depth prior [1, 1, H, W] with values in [0, 1]
        """
        # Depth increases towards bottom of image (closer)
        y_depth = torch.linspace(0.0, 1.0, height).unsqueeze(1).expand(height, width)
        return y_depth.unsqueeze(0).unsqueeze(0)
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Apply spatial priors to feature map.
        
        Args:
            features: Input features [B, C, H, W] where C must equal self.dim
            
        Returns:
            Features with spatial encoding [B, C, H, W]
        """
        B, C, H, W = features.shape
        
        if C != self.dim:
            raise ValueError(f"Feature channels {C} must match dim {self.dim}")
        
        if H > self.max_height or W > self.max_width:
            raise ValueError(
                f"Spatial size ({H}, {W}) exceeds max ({self.max_height}, {self.max_width})"
            )
        
        # Apply rotary position encoding
        features = self.rotary_2d(features)
        
        # Apply perspective scale modulation if enabled
        if self.apply_perspective:
            # Interpolate scale prior if sizes don't match
            if H != self.max_height or W != self.max_width:
                scale = F.interpolate(
                    self.scale_prior, 
                    size=(H, W), 
                    mode='bilinear', 
                    align_corners=False
                )
            else:
                scale = self.scale_prior[:, :, :H, :W]
            
            features = features * scale
        
        return features
    
    def get_depth_prior(self, height: int, width: int) -> torch.Tensor:
        """
        Get depth prior for given spatial size.
        
        Args:
            height: Target height
            width: Target width
            
        Returns:
            Depth prior [1, 1, H, W]
        """
        if height != self.max_height or width != self.max_width:
            return F.interpolate(
                self.depth_prior,
                size=(height, width),
                mode='bilinear',
                align_corners=False
            )
        return self.depth_prior


class OcclusionPrior(nn.Module):
    """
    Occlusion reasoning prior.
    
    Encodes the principle that closer objects block farther objects.
    This is used to determine which features should "win" when
    multiple objects overlap at the same spatial location.
    """
    
    def __init__(self, dim: int) -> None:
        """
        Initialize occlusion prior.
        
        Args:
            dim: Feature dimension
        """
        super().__init__()
        self.dim = dim
        
        # Depth-based attention: learns to weight features by estimated depth
        self.depth_attention = nn.Sequential(
            nn.Conv2d(dim, dim // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 4, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        
    def forward(
        self, 
        features: torch.Tensor, 
        depth_prior: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply occlusion-aware weighting to features.
        
        Args:
            features: Input features [B, C, H, W]
            depth_prior: Expected depth [B, 1, H, W] or [1, 1, H, W]
            
        Returns:
            Occlusion-weighted features [B, C, H, W]
        """
        # Estimate local depth confidence from features
        depth_confidence = self.depth_attention(features)  # [B, 1, H, W]
        
        # Combine with prior depth
        combined_depth = depth_confidence * depth_prior
        
        # Weight features by depth (closer = higher weight)
        weighted_features = features * combined_depth
        
        return weighted_features


class CenterSurroundPrior(nn.Module):
    """
    Center-surround spatial attention prior.
    
    The human visual system has a strong center bias - we attend more
    to the center of the visual field. This is largely innate.
    """
    
    def __init__(self, height: int, width: int, sigma: float = 0.3) -> None:
        """
        Initialize center-surround prior.
        
        Args:
            height: Spatial height
            width: Spatial width
            sigma: Spread of center bias (fraction of image size)
        """
        super().__init__()
        
        # Create 2D Gaussian centered in image
        y = torch.linspace(-1, 1, height).unsqueeze(1).expand(height, width)
        x = torch.linspace(-1, 1, width).unsqueeze(0).expand(height, width)
        
        # Gaussian falloff from center
        center_bias = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
        
        # Normalize to [0, 1] range
        center_bias = center_bias / center_bias.max()
        
        self.register_buffer('center_bias', center_bias.unsqueeze(0).unsqueeze(0))
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Apply center-surround attention bias.
        
        Args:
            features: Input features [B, C, H, W]
            
        Returns:
            Center-biased features [B, C, H, W]
        """
        B, C, H, W = features.shape
        
        # Interpolate if sizes don't match
        if H != self.center_bias.shape[2] or W != self.center_bias.shape[3]:
            bias = F.interpolate(
                self.center_bias,
                size=(H, W),
                mode='bilinear',
                align_corners=False
            )
        else:
            bias = self.center_bias
        
        return features * bias


class GridCellPrior(nn.Module):
    """
    Grid cell-like spatial encoding.
    
    Inspired by grid cells in the entorhinal cortex, which encode
    position using periodic (grid-like) firing patterns.
    
    This provides a different type of position encoding that is
    more suitable for spatial navigation and metric distance encoding.
    """
    
    def __init__(
        self,
        dim: int,
        max_height: int,
        max_width: int,
        num_scales: int = 4,
        base_frequency: float = 0.1,
    ) -> None:
        """
        Initialize grid cell prior.
        
        Args:
            dim: Output dimension (must be divisible by 2 * num_scales)
            max_height: Maximum spatial height
            max_width: Maximum spatial width
            num_scales: Number of different grid scales
            base_frequency: Base frequency for smallest grid
        """
        super().__init__()
        
        assert dim % (2 * num_scales) == 0, \
            f"dim ({dim}) must be divisible by 2 * num_scales ({2 * num_scales})"
        
        self.dim = dim
        self.num_scales = num_scales
        self.features_per_scale = dim // num_scales
        
        # Create grid patterns at different scales and orientations
        grid_patterns = self._create_grid_patterns(
            max_height, max_width, num_scales, base_frequency
        )
        self.register_buffer('grid_patterns', grid_patterns)
        
    def _create_grid_patterns(
        self,
        height: int,
        width: int,
        num_scales: int,
        base_freq: float,
    ) -> torch.Tensor:
        """
        Create multi-scale grid patterns.
        
        Returns:
            Grid patterns [dim, H, W]
        """
        y = torch.linspace(0, height, height).unsqueeze(1).expand(height, width)
        x = torch.linspace(0, width, width).unsqueeze(0).expand(height, width)
        
        patterns = []
        features_per_scale = self.features_per_scale
        
        for scale_idx in range(num_scales):
            freq = base_freq * (2 ** scale_idx)
            
            # Multiple orientations per scale
            num_orientations = features_per_scale // 2
            for orient_idx in range(num_orientations):
                angle = orient_idx * math.pi / num_orientations
                
                # Rotated coordinates
                x_rot = x * math.cos(angle) + y * math.sin(angle)
                y_rot = -x * math.sin(angle) + y * math.cos(angle)
                
                # Sinusoidal patterns (like grid cell firing)
                pattern_sin = torch.sin(2 * math.pi * freq * x_rot)
                pattern_cos = torch.cos(2 * math.pi * freq * y_rot)
                
                patterns.append(pattern_sin)
                patterns.append(pattern_cos)
        
        return torch.stack(patterns, dim=0)  # [dim, H, W]
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Add grid cell encoding to features.
        
        Args:
            features: Input features [B, C, H, W]
            
        Returns:
            Features with grid encoding [B, C + dim, H, W]
        """
        B, C, H, W = features.shape
        
        # Interpolate grid patterns if sizes don't match
        if H != self.grid_patterns.shape[1] or W != self.grid_patterns.shape[2]:
            grid = F.interpolate(
                self.grid_patterns.unsqueeze(0),
                size=(H, W),
                mode='bilinear',
                align_corners=False
            ).expand(B, -1, -1, -1)
        else:
            grid = self.grid_patterns.unsqueeze(0).expand(B, -1, -1, -1)
        
        # Concatenate with features
        return torch.cat([features, grid], dim=1)
