"""
Temporal World Model.

Processes sequences of multi-modal inputs to build a unified world state.
Uses temporal priors (causality, decay, periodicity) and transformer
architecture to understand sequences over time.

This is the "temporal cortex" - it integrates information across time
to build a coherent understanding of what's happening.
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

from src.priors.temporal_prior import TemporalPrior


@dataclass
class TemporalWorldModelConfig:
    """Configuration for temporal world model."""
    dim: int = 512
    num_heads: int = 8
    num_layers: int = 6
    max_seq_len: int = 64
    state_dim: int = 256
    dropout: float = 0.1
    ff_dim: Optional[int] = None  # Defaults to 4 * dim
    use_causal: bool = True


class TemporalWorldModel(nn.Module):
    """
    Processes sequences of multi-modal inputs to build a world state.
    
    Pipeline:
    1. Apply temporal prior (position encoding, causality)
    2. Transformer encoder over sequence
    3. Aggregate to world state (optional)
    
    The model can predict future states through the dynamics predictor.
    """
    
    def __init__(self, config: TemporalWorldModelConfig) -> None:
        super().__init__()
        
        self.config = config
        
        # Temporal prior (position encoding, causal mask)
        self.temporal_prior = TemporalPrior(
            max_seq_len=config.max_seq_len,
            dim=config.dim,
        )
        
        # Temporal transformer
        ff_dim = config.ff_dim or config.dim * 4
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.dim,
            nhead=config.num_heads,
            dim_feedforward=ff_dim,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-norm for stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
        )
        
        # State aggregator: sequence -> single state vector
        self.state_aggregator = nn.Sequential(
            nn.Linear(config.dim, config.dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dim, config.state_dim),
        )
        
        # Output normalization
        self.output_norm = nn.LayerNorm(config.dim)
    
    def forward(
        self, 
        sequence: torch.Tensor,
        return_full_sequence: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process sequence to build world state.
        
        Args:
            sequence: Fused multi-modal features [B, T, D]
            return_full_sequence: If True, return all timesteps
            
        Returns:
            Tuple of:
            - World state [B, state_dim] (aggregated)
            - Temporal features [B, T, D] (full sequence)
        """
        B, T, D = sequence.shape
        
        if T > self.config.max_seq_len:
            raise ValueError(
                f"Sequence length {T} exceeds max {self.config.max_seq_len}"
            )
        
        # Apply temporal prior (adds position encoding)
        sequence_with_time, causal_mask = self.temporal_prior(sequence)
        
        # Convert bool mask to float mask for transformer
        # True = masked (don't attend), so we use -inf for True positions
        if self.config.use_causal and causal_mask is not None:
            attn_mask = causal_mask.float().masked_fill(
                causal_mask, float('-inf')
            ).to(sequence.device)
        else:
            attn_mask = None
        
        # Temporal encoding
        encoded = self.transformer(
            sequence_with_time,
            mask=attn_mask,
        )  # [B, T, D]
        
        encoded = self.output_norm(encoded)
        
        # Aggregate to world state (use last token or weighted average)
        # Using last token (most recent, contains all past context due to causal attention)
        last_token = encoded[:, -1, :]  # [B, D]
        world_state = self.state_aggregator(last_token)  # [B, state_dim]
        
        return world_state, encoded
    
    def forward_step(
        self,
        current_features: torch.Tensor,
        past_encoded: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process single timestep incrementally.
        
        Args:
            current_features: Current frame features [B, D]
            past_encoded: Previously encoded sequence [B, T, D]
            
        Returns:
            Tuple of:
            - Current world state [B, state_dim]
            - Updated encoded sequence [B, T+1, D]
        """
        B, D = current_features.shape
        
        # Add sequence dimension
        current = current_features.unsqueeze(1)  # [B, 1, D]
        
        if past_encoded is not None:
            # Concatenate with past
            sequence = torch.cat([past_encoded, current], dim=1)  # [B, T+1, D]
        else:
            sequence = current  # [B, 1, D]
        
        # Process full sequence
        world_state, encoded = self.forward(sequence)
        
        return world_state, encoded


class TemporalAttentionPooling(nn.Module):
    """
    Attention-based temporal pooling.
    
    Instead of using the last token or mean pooling,
    learn to weight timesteps based on their relevance.
    """
    
    def __init__(self, dim: int, num_heads: int = 4) -> None:
        super().__init__()
        
        # Learnable query for pooling
        self.query = nn.Parameter(torch.randn(1, 1, dim))
        
        # Attention
        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True,
        )
        
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """
        Pool sequence using attention.
        
        Args:
            sequence: Input sequence [B, T, D]
            
        Returns:
            Pooled representation [B, D]
        """
        B = sequence.shape[0]
        
        # Expand query for batch
        query = self.query.expand(B, -1, -1)  # [B, 1, D]
        
        # Attend to sequence
        pooled, _ = self.attention(query, sequence, sequence)  # [B, 1, D]
        
        pooled = self.norm(pooled.squeeze(1))  # [B, D]
        
        return pooled


class RecurrentWorldModel(nn.Module):
    """
    Alternative temporal world model using recurrence (GRU).
    
    More efficient for online processing where we receive
    one frame at a time.
    """
    
    def __init__(self, config: TemporalWorldModelConfig) -> None:
        super().__init__()
        
        self.config = config
        
        # GRU for temporal processing
        self.gru = nn.GRU(
            input_size=config.dim,
            hidden_size=config.dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0,
        )
        
        # State output
        self.state_proj = nn.Sequential(
            nn.LayerNorm(config.dim),
            nn.Linear(config.dim, config.state_dim),
        )
    
    def forward(
        self,
        sequence: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process sequence with GRU.
        
        Args:
            sequence: Input sequence [B, T, D]
            hidden: Previous hidden state [num_layers, B, D]
            
        Returns:
            Tuple of:
            - World state [B, state_dim]
            - Full output [B, T, D]
            - Hidden state [num_layers, B, D]
        """
        output, hidden = self.gru(sequence, hidden)  # [B, T, D], [L, B, D]
        
        # Use last output for state
        last_output = output[:, -1, :]  # [B, D]
        world_state = self.state_proj(last_output)  # [B, state_dim]
        
        return world_state, output, hidden
    
    def forward_step(
        self,
        current: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process single timestep.
        
        Args:
            current: Current features [B, D]
            hidden: Previous hidden state
            
        Returns:
            World state and updated hidden
        """
        current = current.unsqueeze(1)  # [B, 1, D]
        output, hidden = self.gru(current, hidden)
        world_state = self.state_proj(output.squeeze(1))
        return world_state, hidden
    
    def get_initial_hidden(
        self, 
        batch_size: int, 
        device: torch.device
    ) -> torch.Tensor:
        """Get initial hidden state."""
        return torch.zeros(
            self.config.num_layers,
            batch_size,
            self.config.dim,
            device=device,
        )
