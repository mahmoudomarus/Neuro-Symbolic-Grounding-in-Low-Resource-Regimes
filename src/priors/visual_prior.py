"""
Visual frequency prior: Gabor filterbank mimicking retinal/V1 processing.

Biological inspiration:
- Retinal ganglion cells have center-surround receptive fields
- V1 contains orientation-selective columns (discovered by Hubel & Wiesel, 1962)
- These structures are innate, not learned - they evolved to detect edges, textures, motion

Implementation:
- Fixed Gabor filters at multiple scales and orientations
- Applied as first layer (before any learning)
- Output represents "what the visual system sees" before higher-level processing
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GaborFilterbank(nn.Module):
    """
    Fixed Gabor filterbank for visual decomposition.
    
    Creates filters that respond to:
    - Different spatial frequencies (scales)
    - Different orientations (angles)
    - Both even (cosine) and odd (sine) phases
    
    These are analogous to simple cells in primary visual cortex.
    """
    
    def __init__(
        self,
        num_orientations: int = 8,
        num_scales: int = 5,
        kernel_size: int = 31,
        lambda_min: float = 4.0,
        lambda_max: float = 64.0,
    ):
        """
        Initialize Gabor filterbank.
        
        Args:
            num_orientations: Number of orientations (default 8 = every 22.5 degrees)
            num_scales: Number of spatial frequency scales
            kernel_size: Size of filter kernels (should be odd)
            lambda_min: Minimum wavelength (highest frequency)
            lambda_max: Maximum wavelength (lowest frequency)
        """
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        
        self.num_orientations = num_orientations
        self.num_scales = num_scales
        self.kernel_size = kernel_size
        self.num_filters = num_orientations * num_scales * 2  # x2 for even/odd phases
        
        # Generate filters
        filters = self._create_filters(lambda_min, lambda_max)
        
        # Register as non-learnable buffer
        self.register_buffer("filters", filters)
        
    def _create_filters(
        self,
        lambda_min: float,
        lambda_max: float,
    ) -> torch.Tensor:
        """Create all Gabor filters."""
        filters = []
        
        # Wavelengths across scales (geometric progression)
        wavelengths = torch.logspace(
            math.log10(lambda_min),
            math.log10(lambda_max),
            self.num_scales
        )
        
        # Orientations (evenly spaced 0 to pi)
        orientations = torch.linspace(0, math.pi, self.num_orientations + 1)[:-1]
        
        # Create coordinate grid
        half_size = self.kernel_size // 2
        x = torch.arange(-half_size, half_size + 1, dtype=torch.float32)
        y = torch.arange(-half_size, half_size + 1, dtype=torch.float32)
        X, Y = torch.meshgrid(x, y, indexing='xy')
        
        for wavelength in wavelengths:
            for orientation in orientations:
                # Rotate coordinates
                X_rot = X * math.cos(orientation) + Y * math.sin(orientation)
                Y_rot = -X * math.sin(orientation) + Y * math.cos(orientation)
                
                # Gaussian envelope
                sigma = wavelength / math.sqrt(2 * math.log(2))
                gaussian = torch.exp(-(X**2 + Y**2) / (2 * sigma**2))
                
                # Even (cosine) phase - symmetric
                gabor_even = gaussian * torch.cos(2 * math.pi * X_rot / wavelength)
                filters.append(gabor_even)
                
                # Odd (sine) phase - antisymmetric
                gabor_odd = gaussian * torch.sin(2 * math.pi * X_rot / wavelength)
                filters.append(gabor_odd)
        
        # Stack and normalize
        filters = torch.stack(filters)  # [num_filters, kernel_size, kernel_size]
        
        # Normalize each filter to have unit norm
        filters = filters / (filters.abs().sum(dim=(1, 2), keepdim=True) + 1e-8)
        
        return filters
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Apply Gabor filterbank to images.
        
        Args:
            images: [B, C, H, W] where C is typically 3 (RGB) or 1 (grayscale)
            
        Returns:
            Filter responses: [B, num_filters*C, H, W]
        """
        B, C, H, W = images.shape
        
        # Reshape filters for group convolution: [num_filters, 1, kernel_size, kernel_size]
        filters = self.filters.unsqueeze(1)
        
        # Reshape input: treat channels separately
        # [B*C, 1, H, W]
        images_reshaped = images.view(B * C, 1, H, W)
        
        # Apply each filter to each channel
        # Expand filters for all channels
        filters_expanded = filters.repeat(C, 1, 1, 1)
        
        # Group convolution: each filter applied to each channel separately
        responses = F.conv2d(
            images_reshaped,
            filters_expanded,
            padding=self.kernel_size // 2,
            groups=C
        )
        
        # Reshape back: [B, C*num_filters, H, W]
        responses = responses.view(B, C * self.num_filters, H, W)
        
        return responses
    
    def get_filter_bank_visualization(self) -> torch.Tensor:
        """Return filters for visualization: [num_filters, kernel_size, kernel_size]."""
        return self.filters


def apply_gabor_filters(
    images: torch.Tensor,
    num_orientations: int = 8,
    num_scales: int = 5,
) -> torch.Tensor:
    """
    Convenience function to apply Gabor filters without explicit class instantiation.
    
    Args:
        images: [B, C, H, W]
        num_orientations: Number of orientations (default 8)
        num_scales: Number of spatial scales (default 5)
        
    Returns:
        Filtered images: [B, C*num_orientations*num_scales*2, H, W]
    """
    # Create filterbank (will be cached if reused)
    gabor = GaborFilterbank(
        num_orientations=num_orientations,
        num_scales=num_scales,
    )
    
    # Move to same device as images
    gabor = gabor.to(images.device)
    
    return gabor(images)
