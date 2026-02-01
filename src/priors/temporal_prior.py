"""
Temporal innate priors - understanding time and causality.

These encode fundamental temporal reasoning that appears to be innate:

1. TemporalPrior: Core temporal processing
   - Causality (future cannot affect past)
   - Temporal decay (recent events more relevant)
   - Sinusoidal position encoding (sense of time)

2. CausalMask: Enforces causal attention
3. TemporalDecay: Exponential recency weighting
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalPrior(nn.Module):
    """
    Innate temporal processing.
    
    Encodes fundamental properties of time:
    - Causality: Future cannot affect past (enforced via causal mask)
    - Recency: Recent events are more relevant (exponential decay)
    - Periodicity: Sense of time via sinusoidal encoding
    
    This is the "firmware" of temporal perception - it exists before learning.
    """
    
    def __init__(
        self, 
        max_seq_len: int, 
        dim: int,
        decay_rate: float = 0.1,
    ) -> None:
        """
        Initialize temporal prior.
        
        Args:
            max_seq_len: Maximum sequence length
            dim: Feature dimension
            decay_rate: Rate of exponential temporal decay
        """
        super().__init__()
        
        self.max_seq_len = max_seq_len
        self.dim = dim
        self.decay_rate = decay_rate
        
        # Causal mask: upper triangular = True (masked out)
        # This enforces that position i can only attend to positions <= i
        causal_mask = torch.triu(
            torch.ones(max_seq_len, max_seq_len, dtype=torch.bool), 
            diagonal=1
        )
        self.register_buffer('causal_mask', causal_mask)
        
        # Temporal decay: exponential weighting favoring recent events
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        decay = torch.exp(-decay_rate * positions)
        self.register_buffer('temporal_decay', decay)
        
        # Sinusoidal position encoding (innate sense of time)
        position_encoding = self._create_sinusoidal_encoding(max_seq_len, dim)
        self.register_buffer('position_encoding', position_encoding)
        
    def _create_sinusoidal_encoding(self, max_len: int, dim: int) -> torch.Tensor:
        """
        Create sinusoidal position encoding.
        
        This is based on the transformer positional encoding,
        which uses sin/cos at different frequencies to encode position.
        
        Returns:
            Position encoding [1, max_len, dim]
        """
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        
        # Create frequency bands
        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) * 
            (-math.log(10000.0) / dim)
        )
        
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)  # [1, max_len, dim]
    
    def forward(
        self, 
        x: torch.Tensor,
        return_mask: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply temporal prior to sequence.
        
        Args:
            x: Input sequence [B, T, D]
            return_mask: Whether to return causal mask
            
        Returns:
            Tuple of:
            - Sequence with temporal encoding [B, T, D]
            - Causal mask [T, T] (if return_mask=True)
        """
        B, T, D = x.shape
        
        if T > self.max_seq_len:
            raise ValueError(f"Sequence length {T} exceeds max {self.max_seq_len}")
        
        if D != self.dim:
            raise ValueError(f"Feature dim {D} must match prior dim {self.dim}")
        
        # Add position encoding
        x_with_time = x + self.position_encoding[:, :T, :]
        
        if return_mask:
            mask = self.causal_mask[:T, :T]
            return x_with_time, mask
        
        return x_with_time, None
    
    def get_decay_weights(self, seq_len: int) -> torch.Tensor:
        """
        Get temporal decay weights for sequence.
        
        Args:
            seq_len: Sequence length
            
        Returns:
            Decay weights [seq_len] with most recent = highest weight
        """
        # Reverse so most recent (last) has highest weight
        return self.temporal_decay[:seq_len].flip(0)
    
    def apply_decay(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply temporal decay weighting to sequence.
        
        Args:
            x: Input sequence [B, T, D]
            
        Returns:
            Decay-weighted sequence [B, T, D]
        """
        T = x.shape[1]
        weights = self.get_decay_weights(T).unsqueeze(0).unsqueeze(-1)  # [1, T, 1]
        return x * weights


class RelativeTemporalEncoding(nn.Module):
    """
    Relative temporal position encoding.
    
    Instead of absolute positions, encodes relative time differences.
    This is useful for attention mechanisms where we care about
    "how long ago" rather than "which position".
    """
    
    def __init__(self, max_relative_positions: int, num_heads: int) -> None:
        """
        Initialize relative temporal encoding.
        
        Args:
            max_relative_positions: Maximum relative distance to encode
            num_heads: Number of attention heads
        """
        super().__init__()
        
        self.max_relative_positions = max_relative_positions
        self.num_heads = num_heads
        
        # Learnable relative position embeddings
        # Range: [-max_rel, +max_rel] -> [0, 2*max_rel]
        num_embeddings = 2 * max_relative_positions + 1
        self.relative_embeddings = nn.Embedding(num_embeddings, num_heads)
        
    def forward(self, seq_len: int) -> torch.Tensor:
        """
        Get relative position bias matrix.
        
        Args:
            seq_len: Sequence length
            
        Returns:
            Relative position bias [num_heads, seq_len, seq_len]
        """
        # Create relative position indices
        positions = torch.arange(seq_len)
        relative_positions = positions.unsqueeze(1) - positions.unsqueeze(0)  # [T, T]
        
        # Clip to valid range
        relative_positions = relative_positions.clamp(
            -self.max_relative_positions, 
            self.max_relative_positions
        )
        
        # Shift to positive indices
        relative_positions = relative_positions + self.max_relative_positions
        
        # Get embeddings
        bias = self.relative_embeddings(relative_positions)  # [T, T, num_heads]
        
        return bias.permute(2, 0, 1)  # [num_heads, T, T]


class TemporalConvolutionPrior(nn.Module):
    """
    Temporal convolution prior.
    
    Provides local temporal context through 1D convolutions.
    This captures the idea that temporal perception involves
    integration over short time windows.
    """
    
    def __init__(
        self,
        dim: int,
        kernel_sizes: Tuple[int, ...] = (3, 5, 7),
    ) -> None:
        """
        Initialize temporal convolution prior.
        
        Args:
            dim: Feature dimension
            kernel_sizes: Tuple of kernel sizes for multi-scale temporal context
        """
        super().__init__()
        
        self.convs = nn.ModuleList([
            nn.Conv1d(
                dim, dim // len(kernel_sizes),
                kernel_size=k,
                padding=k // 2,
                groups=dim // len(kernel_sizes),  # Depthwise
            )
            for k in kernel_sizes
        ])
        
        self.output_proj = nn.Linear(dim, dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply multi-scale temporal convolution.
        
        Args:
            x: Input sequence [B, T, D]
            
        Returns:
            Convolved sequence [B, T, D]
        """
        # Reshape for conv1d: [B, D, T]
        x_conv = x.permute(0, 2, 1)
        
        # Apply convolutions at different scales
        outputs = []
        for conv in self.convs:
            outputs.append(conv(x_conv))
        
        # Concatenate and reshape back
        combined = torch.cat(outputs, dim=1)  # [B, D, T]
        combined = combined.permute(0, 2, 1)  # [B, T, D]
        
        return self.output_proj(combined)


class CausalityPrior(nn.Module):
    """
    Explicit causality prior for attention.
    
    Generates causal masks and can be used to enforce
    that information only flows forward in time.
    """
    
    def __init__(self, max_seq_len: int) -> None:
        """
        Initialize causality prior.
        
        Args:
            max_seq_len: Maximum sequence length
        """
        super().__init__()
        
        # Standard causal mask
        causal = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1)
        self.register_buffer('causal_mask', causal.bool())
        
        # Lookahead mask (for bidirectional with limited future context)
        # Can see k steps into future
        
    def get_causal_mask(
        self, 
        seq_len: int,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Get causal attention mask.
        
        Args:
            seq_len: Sequence length
            device: Target device
            
        Returns:
            Causal mask [seq_len, seq_len] where True = masked
        """
        mask = self.causal_mask[:seq_len, :seq_len]
        if device is not None:
            mask = mask.to(device)
        return mask
    
    def get_lookahead_mask(
        self, 
        seq_len: int, 
        lookahead: int = 1
    ) -> torch.Tensor:
        """
        Get mask that allows limited lookahead.
        
        Args:
            seq_len: Sequence length
            lookahead: Number of future positions to allow
            
        Returns:
            Lookahead mask [seq_len, seq_len]
        """
        # Allow attending to current + lookahead future positions
        mask = torch.triu(
            torch.ones(seq_len, seq_len), 
            diagonal=lookahead + 1
        )
        return mask.bool()


class RhythmPrior(nn.Module):
    """
    Rhythm and periodicity detection prior.
    
    Encodes the innate ability to detect periodic patterns,
    which is fundamental to music and speech perception.
    """
    
    def __init__(
        self,
        dim: int,
        periods: Tuple[int, ...] = (2, 3, 4, 8, 16),
    ) -> None:
        """
        Initialize rhythm prior.
        
        Args:
            dim: Feature dimension
            periods: Tuple of periods to detect
        """
        super().__init__()
        
        self.periods = periods
        
        # Create periodic kernels
        kernels = []
        for period in periods:
            kernel = torch.zeros(1, 1, period * 2)
            # Create "on-beat" pattern
            for i in range(0, period * 2, period):
                kernel[0, 0, i] = 1.0
            kernels.append(kernel)
        
        self.register_buffer('kernels', torch.cat(kernels, dim=0))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Detect rhythmic patterns in sequence.
        
        Args:
            x: Input sequence [B, T, D] or [B, T]
            
        Returns:
            Rhythm detection for each period [B, len(periods), T]
        """
        if x.dim() == 3:
            # Average across features
            x = x.mean(dim=-1)  # [B, T]
        
        # Add channel dim for conv1d
        x = x.unsqueeze(1)  # [B, 1, T]
        
        # Detect each period
        outputs = []
        offset = 0
        for i, period in enumerate(self.periods):
            kernel = self.kernels[i:i+1, :, :]  # [1, 1, period*2]
            out = F.conv1d(x, kernel, padding=period)
            outputs.append(out)
        
        return torch.cat(outputs, dim=1)  # [B, len(periods), T]
