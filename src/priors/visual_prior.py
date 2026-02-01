"""
Visual innate priors - structures that exist in human vision from birth.

These are NOT learned features but mathematically defined transformations
that mimic the innate processing in the human visual system:

1. ColorOpponencyPrior: RGB -> Opponent color channels (L, RG, BY)
   - Based on retinal ganglion cell responses
   
2. GaborPrior: Oriented edge detection at multiple scales
   - Based on V1 simple cell receptive fields
   
3. DepthCuesPrior: Monocular depth cues
   - Vertical position (lower = closer for ground plane)
   - Texture gradient (more detail = closer)
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ColorOpponencyPrior(nn.Module):
    """
    Transform RGB to opponent color space (innate, not learned).
    
    This is how human retinal ganglion cells actually encode color.
    The opponent channels are:
    - L (Luminance): Brightness channel, 0.5*R + 0.5*G
    - RG (Red-Green): Color opponent, R - G
    - BY (Blue-Yellow): Color opponent, 0.5*(R + G) - B
    
    This transformation is biologically grounded - it's how color is encoded
    before it reaches the visual cortex.
    """
    
    def __init__(self) -> None:
        super().__init__()
        # Fixed transformation matrix (not learnable)
        # Each row defines how to combine RGB for one opponent channel
        transform = torch.tensor([
            [0.5, 0.5, 0.0],      # L channel (luminance)
            [1.0, -1.0, 0.0],     # RG channel (red-green opponent)
            [0.5, 0.5, -1.0],     # BY channel (blue-yellow opponent)
        ], dtype=torch.float32)
        self.register_buffer('transform', transform)
    
    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        Transform RGB image to opponent color space.
        
        Args:
            rgb: Input tensor [B, 3, H, W] in RGB format, values in [0, 1] or normalized
            
        Returns:
            Opponent color tensor [B, 3, H, W] with channels [L, RG, BY]
        """
        B, C, H, W = rgb.shape
        if C != 3:
            raise ValueError(f"Expected 3 channels (RGB), got {C}")
        
        # Reshape for matrix multiplication: [B, H, W, 3]
        rgb_hwc = rgb.permute(0, 2, 3, 1)
        
        # Apply transformation: [B, H, W, 3] @ [3, 3].T -> [B, H, W, 3]
        opponent_hwc = torch.matmul(rgb_hwc, self.transform.T)
        
        # Reshape back to [B, 3, H, W]
        opponent = opponent_hwc.permute(0, 3, 1, 2)
        
        return opponent
    
    def inverse(self, opponent: torch.Tensor) -> torch.Tensor:
        """
        Transform opponent color space back to RGB (approximate inverse).
        
        Args:
            opponent: Opponent tensor [B, 3, H, W] with channels [L, RG, BY]
            
        Returns:
            RGB tensor [B, 3, H, W]
        """
        # Compute pseudo-inverse of transform matrix
        transform_inv = torch.linalg.pinv(self.transform)
        
        B, C, H, W = opponent.shape
        opponent_hwc = opponent.permute(0, 2, 3, 1)
        rgb_hwc = torch.matmul(opponent_hwc, transform_inv.T)
        rgb = rgb_hwc.permute(0, 3, 1, 2)
        
        return rgb


class GaborPrior(nn.Module):
    """
    Gabor filters at multiple orientations and scales.
    
    These filters exist in primary visual cortex (V1) from birth.
    They are NOT learned - they are mathematically defined to detect
    edges and textures at different orientations and spatial frequencies.
    
    Gabor filters are the product of a Gaussian envelope and a sinusoidal wave,
    which models the receptive fields of simple cells in V1.
    """
    
    def __init__(
        self, 
        num_orientations: int = 8, 
        num_scales: int = 4, 
        kernel_size: int = 15
    ) -> None:
        """
        Initialize Gabor filter bank.
        
        Args:
            num_orientations: Number of orientation bins (default 8 = 22.5° steps)
            num_scales: Number of spatial frequency scales
            kernel_size: Size of Gabor kernels (must be odd)
        """
        super().__init__()
        
        if kernel_size % 2 == 0:
            kernel_size += 1  # Ensure odd size for symmetric filters
            
        self.num_orientations = num_orientations
        self.num_scales = num_scales
        self.kernel_size = kernel_size
        self.num_filters = num_orientations * num_scales
        
        # Create Gabor filter bank
        filters = self._create_gabor_bank()
        self.register_buffer('filters', filters)  # [N_filters, 1, K, K]
        
    def _create_gabor_kernel(
        self, 
        sigma: float, 
        theta: float, 
        lambd: float, 
        gamma: float = 0.5,
        psi: float = 0.0
    ) -> torch.Tensor:
        """
        Create a single Gabor kernel.
        
        Args:
            sigma: Standard deviation of Gaussian envelope
            theta: Orientation in radians
            lambd: Wavelength of sinusoidal factor
            gamma: Spatial aspect ratio (ellipticity)
            psi: Phase offset
            
        Returns:
            Gabor kernel tensor [K, K]
        """
        ksize = self.kernel_size
        half = ksize // 2
        
        # Create coordinate grids
        y, x = torch.meshgrid(
            torch.arange(-half, half + 1, dtype=torch.float32),
            torch.arange(-half, half + 1, dtype=torch.float32),
            indexing='ij'
        )
        
        # Rotate coordinates
        x_theta = x * math.cos(theta) + y * math.sin(theta)
        y_theta = -x * math.sin(theta) + y * math.cos(theta)
        
        # Gabor function: Gaussian * Sinusoid
        gaussian = torch.exp(-0.5 * (x_theta**2 + gamma**2 * y_theta**2) / sigma**2)
        sinusoid = torch.cos(2 * math.pi * x_theta / lambd + psi)
        
        gabor = gaussian * sinusoid
        
        # Normalize to zero mean and unit L2 norm
        gabor = gabor - gabor.mean()
        gabor = gabor / (gabor.norm() + 1e-8)
        
        return gabor
    
    def _create_gabor_bank(self) -> torch.Tensor:
        """
        Create bank of Gabor filters at multiple orientations and scales.
        
        Returns:
            Filter bank tensor [N_filters, 1, K, K]
        """
        filters = []
        
        for scale_idx in range(self.num_scales):
            # Sigma and wavelength increase with scale
            sigma = 2.0 + scale_idx * 1.5
            lambd = 4.0 + scale_idx * 2.0
            
            for orient_idx in range(self.num_orientations):
                # Orientation from 0 to pi
                theta = orient_idx * math.pi / self.num_orientations
                
                gabor = self._create_gabor_kernel(sigma, theta, lambd)
                filters.append(gabor.unsqueeze(0).unsqueeze(0))
        
        return torch.cat(filters, dim=0)  # [N, 1, K, K]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Gabor filter bank to input.
        
        Args:
            x: Input tensor [B, 1, H, W] (single channel, e.g., luminance)
            
        Returns:
            Filtered tensor [B, N_filters, H, W] with edge responses
        """
        B, C, H, W = x.shape
        
        if C != 1:
            # If multi-channel, apply to each channel and concatenate
            outputs = []
            for c in range(C):
                out_c = F.conv2d(
                    x[:, c:c+1, :, :], 
                    self.filters, 
                    padding=self.kernel_size // 2
                )
                outputs.append(out_c)
            return torch.cat(outputs, dim=1)
        
        # Apply all Gabor filters
        return F.conv2d(x, self.filters, padding=self.kernel_size // 2)
    
    def get_filter_visualization(self) -> torch.Tensor:
        """
        Get filter bank for visualization.
        
        Returns:
            Filter bank [num_scales, num_orientations, K, K]
        """
        return self.filters.squeeze(1).reshape(
            self.num_scales, self.num_orientations, 
            self.kernel_size, self.kernel_size
        )


class DepthCuesPrior(nn.Module):
    """
    Innate monocular depth cues.
    
    These are depth perception mechanisms that work with a single eye,
    based on assumptions about the world that are largely innate:
    
    1. Vertical position: Objects lower in the visual field are typically closer
       (assumes a ground plane)
    2. Relative size: Larger retinal image = closer object
    3. Texture gradient: More texture detail visible = closer
    
    This module provides the vertical position prior as a fixed spatial map.
    """
    
    def __init__(self, height: int, width: int) -> None:
        """
        Initialize depth cues prior.
        
        Args:
            height: Expected input height
            width: Expected input width
        """
        super().__init__()
        self.height = height
        self.width = width
        
        # Vertical position prior: bottom of image = close (1.0), top = far (0.0)
        # This encodes the assumption of a ground plane
        y_positions = torch.linspace(0.0, 1.0, height).unsqueeze(1).expand(height, width)
        # Invert so that bottom (higher y index) = closer
        vertical_prior = 1.0 - y_positions
        
        self.register_buffer('vertical_prior', vertical_prior.unsqueeze(0).unsqueeze(0))
        
        # Center-surround prior: objects near image center often closer (attention bias)
        y_grid = torch.linspace(-1, 1, height).unsqueeze(1).expand(height, width)
        x_grid = torch.linspace(-1, 1, width).unsqueeze(0).expand(height, width)
        distance_from_center = torch.sqrt(x_grid**2 + y_grid**2)
        center_prior = 1.0 - (distance_from_center / distance_from_center.max())
        
        self.register_buffer('center_prior', center_prior.unsqueeze(0).unsqueeze(0))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get depth prior map for input.
        
        Args:
            x: Input tensor [B, C, H, W] (used only for batch size and device)
            
        Returns:
            Depth prior map [B, 1, H, W] with values in [0, 1]
            Higher values indicate expected closer depth
        """
        B, C, H, W = x.shape
        
        # Handle size mismatch by interpolating
        if H != self.height or W != self.width:
            vertical = F.interpolate(
                self.vertical_prior, 
                size=(H, W), 
                mode='bilinear', 
                align_corners=False
            )
        else:
            vertical = self.vertical_prior
            
        return vertical.expand(B, -1, -1, -1)
    
    def get_combined_prior(self, x: torch.Tensor, vertical_weight: float = 0.7) -> torch.Tensor:
        """
        Get combined depth prior (vertical + center).
        
        Args:
            x: Input tensor [B, C, H, W]
            vertical_weight: Weight for vertical prior (1 - this for center)
            
        Returns:
            Combined depth prior [B, 1, H, W]
        """
        B, C, H, W = x.shape
        
        # Interpolate if needed
        if H != self.height or W != self.width:
            vertical = F.interpolate(self.vertical_prior, size=(H, W), mode='bilinear', align_corners=False)
            center = F.interpolate(self.center_prior, size=(H, W), mode='bilinear', align_corners=False)
        else:
            vertical = self.vertical_prior
            center = self.center_prior
        
        combined = vertical_weight * vertical + (1 - vertical_weight) * center
        return combined.expand(B, -1, -1, -1)


class TextureGradientPrior(nn.Module):
    """
    Texture gradient depth cue.
    
    Objects closer to the viewer show more texture detail (higher spatial frequency).
    This is computed by measuring local spatial frequency content.
    """
    
    def __init__(self, kernel_size: int = 5) -> None:
        """
        Initialize texture gradient prior.
        
        Args:
            kernel_size: Size of local window for texture analysis
        """
        super().__init__()
        self.kernel_size = kernel_size
        
        # Sobel filters for gradient computation
        sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        sobel_y = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute texture gradient (local spatial frequency content).
        
        Args:
            x: Input tensor [B, 1, H, W] (single channel)
            
        Returns:
            Texture gradient map [B, 1, H, W]
            Higher values indicate more texture (closer depth)
        """
        # Compute gradients
        grad_x = F.conv2d(x, self.sobel_x, padding=1)
        grad_y = F.conv2d(x, self.sobel_y, padding=1)
        
        # Gradient magnitude
        gradient_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
        
        # Local pooling to get texture density
        texture = F.avg_pool2d(
            gradient_mag, 
            kernel_size=self.kernel_size, 
            stride=1, 
            padding=self.kernel_size // 2
        )
        
        # Normalize to [0, 1]
        texture = texture / (texture.max() + 1e-8)
        
        return texture
