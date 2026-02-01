"""
Spatial coordinate prior: 3D rotary position encoding for body-centric coordinates.

Biological inspiration:
- Vestibular system provides innate sense of position and orientation
- Humans know up/down, left/right without learning
- Spatial awareness is fundamental, not derived

Implementation:
- 3D extension of rotary positional encoding
- Encodes (x, y, z) position and (roll, pitch, yaw) orientation
- Fixed frequency basis (not learned)
- Applied to proprioceptive or spatial features
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


class RotaryEmbedding3D(nn.Module):
    """
    Three-dimensional rotary position embedding for spatial coordinates.
    
    Extends rotary position encoding to 3D space (position + orientation).
    Used for encoding body state in 3D world coordinates.
    
    Encodes:
    - Position: (x, y, z) coordinates
    - Orientation: (roll, pitch, yaw) angles
    
    All encoded using frequency-based rotation (RoPE-style).
    """
    
    def __init__(
        self,
        dim: int = 256,
        max_position: float = 100.0,  # meters
        max_angle: float = 2 * math.pi,  # radians
        base_freq: float = 10000.0,
        num_freq_bands: int = 16,
    ):
        """
        Initialize 3D rotary position encoding.
        
        Args:
            dim: Feature dimension (must be divisible by 6 for x,y,z,roll,pitch,yaw)
            max_position: Maximum position magnitude (for position encoding)
            max_angle: Maximum angle (2*pi for full rotation)
            base_freq: Base frequency for RoPE
            num_freq_bands: Number of frequency bands per dimension
        """
        super().__init__()
        
        assert dim % 6 == 0, f"dim must be divisible by 6, got {dim}"
        
        self.dim = dim
        self.max_position = max_position
        self.max_angle = max_angle
        self.base_freq = base_freq
        self.num_freq_bands = num_freq_bands
        
        # Each dimension gets dim/6 channels
        self.dim_per_axis = dim // 6
        
        # Position axes: x, y, z
        # Orientation axes: roll, pitch, yaw
        
        # Create frequency bands for each axis type
        # Positions use different scale than orientations
        self._create_position_frequencies()
        self._create_orientation_frequencies()
        
    def _create_position_frequencies(self):
        """Create frequency bands for position encoding (x, y, z)."""
        # Position frequencies: linear spacing from low to high
        # Normalize positions to [-1, 1] first
        freqs = torch.linspace(1, self.num_freq_bands, self.num_freq_bands)
        
        # Register as buffers (non-learnable)
        for axis in ['x', 'y', 'z']:
            self.register_buffer(f"pos_freq_{axis}", freqs.clone())
    
    def _create_orientation_frequencies(self):
        """Create frequency bands for orientation encoding (roll, pitch, yaw)."""
        # Orientation frequencies: different scale for angular encoding
        freqs = torch.linspace(1, self.num_freq_bands, self.num_freq_bands)
        
        for axis in ['roll', 'pitch', 'yaw']:
            self.register_buffer(f"ori_freq_{axis}", freqs.clone())
    
    def _positional_encoding(self, position: torch.Tensor, freq: torch.Tensor) -> torch.Tensor:
        """
        Generate positional encoding using sinusoidal functions.
        
        Args:
            position: [B, 1] position values
            freq: [num_bands] frequency multipliers
            
        Returns:
            Encoded position: [B, num_bands * 2]
        """
        # Normalize to [-1, 1]
        position_norm = torch.clamp(position / self.max_position, -1, 1)
        
        # Expand: [B, 1, 1] * [1, 1, num_bands] = [B, 1, num_bands]
        angles = position_norm.unsqueeze(-1) * freq.unsqueeze(0).unsqueeze(0)
        
        # Sinusoidal encoding: sin and cos
        sin_enc = torch.sin(angles)
        cos_enc = torch.cos(angles)
        
        # Concatenate: [B, 1, num_bands * 2]
        encoding = torch.cat([sin_enc, cos_enc], dim=-1)
        
        # Flatten: [B, num_bands * 2]
        return encoding.squeeze(1)
    
    def _orientation_encoding(self, angle: torch.Tensor, freq: torch.Tensor) -> torch.Tensor:
        """
        Generate orientation encoding using sinusoidal functions.
        
        Args:
            angle: [B, 1] angle values in radians
            freq: [num_bands] frequency multipliers
            
        Returns:
            Encoded angle: [B, num_bands * 2]
        """
        # Angles wrap around, so we don't normalize
        # Expand: [B, 1, 1] * [1, 1, num_bands] = [B, 1, num_bands]
        angles = angle.unsqueeze(-1) * freq.unsqueeze(0).unsqueeze(0)
        
        # Sinusoidal encoding
        sin_enc = torch.sin(angles)
        cos_enc = torch.cos(angles)
        
        encoding = torch.cat([sin_enc, cos_enc], dim=-1)
        
        return encoding.squeeze(1)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Encode 3D state (position + orientation) into features.
        
        Args:
            state: [B, 12] where:
                - [:3] = position (x, y, z)
                - [3:6] = velocity (vx, vy, vz) - treated like position
                - [6:12] = IMU/orientation (roll, pitch, yaw and derivatives)
                
        Returns:
            Encoded features: [B, dim]
        """
        B = state.shape[0]
        
        # Extract components
        position = state[:, :3]  # [B, 3]
        velocity = state[:, 3:6]  # [B, 3]
        orientation = state[:, 6:9]  # [B, 3] - roll, pitch, yaw
        angular_vel = state[:, 9:12]  # [B, 3]
        
        encodings = []
        
        # Encode position (x, y, z)
        for i, axis in enumerate(['x', 'y', 'z']):
            freq = getattr(self, f"pos_freq_{axis}")
            enc = self._positional_encoding(position[:, i:i+1], freq)
            encodings.append(enc)
        
        # Encode velocity like position
        for i, axis in enumerate(['x', 'y', 'z']):
            freq = getattr(self, f"pos_freq_{axis}")
            enc = self._positional_encoding(velocity[:, i:i+1], freq)
            encodings.append(enc)
        
        # Encode orientation (roll, pitch, yaw)
        for i, axis in enumerate(['roll', 'pitch', 'yaw']):
            freq = getattr(self, f"ori_freq_{axis}")
            enc = self._orientation_encoding(orientation[:, i:i+1], freq)
            encodings.append(enc)
        
        # Encode angular velocity like orientation
        for i, axis in enumerate(['roll', 'pitch', 'yaw']):
            freq = getattr(self, f"ori_freq_{axis}")
            enc = self._orientation_encoding(angular_vel[:, i:i+1], freq)
            encodings.append(enc)
        
        # Concatenate all encodings
        # We have 12 components, each with num_bands * 2 features
        # But we need to match dim, so we'll project down
        all_encodings = torch.cat(encodings, dim=1)  # [B, 12 * num_bands * 2]
        
        # Project to desired dimension
        # For now, just take first 'dim' features
        # In full implementation, we'd learn a projection
        if all_encodings.shape[1] > self.dim:
            return all_encodings[:, :self.dim]
        else:
            # Pad if needed
            padding = torch.zeros(B, self.dim - all_encodings.shape[1], device=state.device)
            return torch.cat([all_encodings, padding], dim=1)


class SimplifiedRotary3D(nn.Module):
    """
    Simplified 3D rotary encoding using standard RoPE extended to 3D.
    
    More efficient than the full positional encoding above.
    """
    
    def __init__(
        self,
        dim: int = 256,
        max_seq_len: int = 2048,
        base: float = 10000.0,
    ):
        super().__init__()
        
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Create inverse frequencies (standard RoPE)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
    def _compute_cos_sin(self, seq_len: int, device: torch.device):
        """Precompute cos and sin for rotary encoding."""
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        
        # Outer product: [seq_len] x [dim//2] -> [seq_len, dim//2]
        freqs = torch.outer(t, self.inv_freq)
        
        # Duplicate for complex representation
        emb = torch.cat([freqs, freqs], dim=-1)  # [seq_len, dim]
        
        cos = emb.cos()
        sin = emb.sin()
        
        return cos, sin
    
    def forward(
        self,
        x: torch.Tensor,
        position_3d: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply rotary encoding to features based on 3D position.
        
        Args:
            x: [B, N, dim] features to encode
            position_3d: [B, N, 3] 3D positions
            
        Returns:
            Encoded features: [B, N, dim]
        """
        B, N, D = x.shape
        
        # Convert 3D position to 1D sequence index (simple sum)
        # More sophisticated: use Hilbert curve or similar
        seq_indices = (position_3d.abs().sum(dim=-1) * 10).long() % self.max_seq_len
        
        # Compute cos/sin
        cos, sin = self._compute_cos_sin(self.max_seq_len, x.device)
        
        # Apply rotary encoding
        # This is a simplified version - full RoPE applies rotation in complex plane
        x_rot = x * cos[seq_indices] + torch.roll(x, shifts=D//2, dims=-1) * sin[seq_indices]
        
        return x_rot


def apply_3d_rotary(
    features: torch.Tensor,
    position_3d: torch.Tensor,
    dim: int = 256,
) -> torch.Tensor:
    """
    Convenience function for 3D rotary encoding.
    
    Args:
        features: [B, N, dim] features
        position_3d: [B, N, 3] 3D positions
        dim: Feature dimension
        
    Returns:
        Rotated features: [B, N, dim]
    """
    rotary = SimplifiedRotary3D(dim=dim)
    rotary = rotary.to(features.device)
    return rotary(features, position_3d)
