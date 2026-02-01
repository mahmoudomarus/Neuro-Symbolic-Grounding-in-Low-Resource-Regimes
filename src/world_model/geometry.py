"""
Innate physics and geometric priors for the world model.

RotaryEmbedding2D encodes (height, width) position as a fundamental truth via
rotation in the latent space, so that "Object A at (0,0)" and "Object A at (1,1)"
are the same concept—just translated. This gives the model innate spatial awareness
without learning position from scratch.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .config import GeometryConfig


def _compute_freqs(dim: int, base: float) -> torch.Tensor:
    """Inverse frequencies for RoPE: 10000^(-2k/dim) for k in 0..dim/2-1."""
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    return inv_freq


class RotaryEmbedding2D(nn.Module):
    """
    Two-dimensional rotary position embedding for spatial feature maps.

    Encodes (row, col) position by rotating channel pairs with angles that depend
    on both coordinates. Same object at different (i, j) gets a consistent
    relative rotation, so the model treats it as the same concept, moved.
    Uses mixed-frequency RoPE: each pair rotates by theta_h * i + theta_w * j.
    """

    def __init__(self, config: GeometryConfig, device: Optional[torch.device] = None) -> None:
        super().__init__()
        self.config = config
        dim = config.dim
        if dim % 2 != 0:
            raise ValueError("RotaryEmbedding2D requires even dim.")
        self.dim = dim
        self.max_height = config.max_height
        self.max_width = config.max_width

        # One set of frequencies per axis: dim/2 values for rotating dim/2 channel pairs
        half = dim // 2
        inv_freq_h = _compute_freqs(dim, config.base_freq)  # length dim/2
        inv_freq_w = _compute_freqs(dim, config.base_freq * config.axis_balance)

        self.register_buffer("inv_freq_h", inv_freq_h)
        self.register_buffer("inv_freq_w", inv_freq_w)

        # Precompute cos/sin for all (i, j) up to max_height, max_width
        self._build_cos_sin_cache(device)

    def _build_cos_sin_cache(self, device: Optional[torch.device] = None) -> None:
        """Build cos/sin cache for positions [0..max_height-1] x [0..max_width-1]."""
        H, W = self.max_height, self.max_width
        half = self.dim // 2

        # Angles: for position (i, j), angle = i * inv_freq_h + j * inv_freq_w
        # inv_freq_h [half], inv_freq_w [half]
        # i [H], j [W] -> i * inv_freq_h: [H, half], j * inv_freq_w: [W, half]
        i = torch.arange(H, dtype=torch.float32, device=device or self.inv_freq_h.device)
        j = torch.arange(W, dtype=torch.float32, device=device or self.inv_freq_w.device)
        angle_h = torch.outer(i, self.inv_freq_h)  # [H, half]
        angle_w = torch.outer(j, self.inv_freq_w)  # [W, half]
        # angle[b, i, j, :] = angle_h[i, :] + angle_w[j, :] -> [H, W, half]
        angle = angle_h.unsqueeze(1) + angle_w.unsqueeze(0)  # [H, W, half]

        cos = angle.cos()  # [H, W, half]
        sin = angle.sin()  # [H, W, half]

        # For rotation we need cos/sin per channel pair: [1, half, H, W]
        cos = cos.permute(2, 0, 1).unsqueeze(0)  # [1, half, H, W]
        sin = sin.permute(2, 0, 1).unsqueeze(0)  # [1, half, H, W]
        self.register_buffer("_cos_cached", cos)
        self.register_buffer("_sin_cached", sin)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply 2D rotary embedding to a spatial map.

        Args:
            x: Spatial map of shape (B, C, H, W), C must match config.dim.

        Returns:
            Same shape (B, C, H, W) with position-dependent rotation applied.
        """
        B, C, H, W = x.shape
        if C != self.dim:
            raise ValueError(f"Channel dim {C} does not match geometry dim {self.dim}.")
        if H > self.max_height or W > self.max_width:
            raise ValueError(
                f"Spatial size ({H}, {W}) exceeds max ({self.max_height}, {self.max_width})."
            )

        cos = self._cos_cached[:, :, :H, :W]   # [1, half, H, W]
        sin = self._sin_cached[:, :, :H, :W]   # [1, half, H, W]

        # Split channels into pairs: x1 [B, half, H, W], x2 [B, half, H, W]
        x1, x2 = x.chunk(2, dim=1)
        # Rotation: (x1, x2) -> (x1*cos - x2*sin, x1*sin + x2*cos)
        x_rotated = torch.cat(
            (x1 * cos - x2 * sin, x1 * sin + x2 * cos),
            dim=1,
        )
        return x_rotated


class GeometricBias(nn.Module):
    """
    Wrapper that applies 2D geometric (rotary) prior to a spatial feature map.

    Use after convolutional layers to inject innate spatial structure so that
    the latent space respects "same concept, different position" equivariance.
    """

    def __init__(self, config: GeometryConfig) -> None:
        super().__init__()
        self.rotary = RotaryEmbedding2D(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply rotary position encoding to the spatial map."""
        return self.rotary(x)
