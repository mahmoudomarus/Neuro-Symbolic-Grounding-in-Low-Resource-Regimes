"""
Cross-modal fusion using attention.

This module fuses multiple modalities (vision, audio, proprioception)
into unified representations using transformer-style attention.

The key insight: cross-modal attention learns which visual features
correspond to which sounds (e.g., a car horn -> the car in the image).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FusionConfig:
    """Configuration for cross-modal fusion."""
    dim: int = 512
    num_heads: int = 8
    num_layers: int = 4
    dropout: float = 0.1
    ff_dim: Optional[int] = None  # Defaults to 4 * dim


class ModalityEmbedding(nn.Module):
    """
    Learnable embeddings to identify modality type.
    
    These embeddings are added to features to help the model
    distinguish between vision, audio, and proprioception tokens.
    """
    
    def __init__(self, dim: int, num_modalities: int = 3) -> None:
        super().__init__()
        
        self.embeddings = nn.Embedding(num_modalities, dim)
        
        # Standard modality indices
        self.VISION = 0
        self.AUDIO = 1
        self.PROPRIO = 2
    
    def forward(self, modality_idx: int, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Get modality embedding for all tokens in a sequence.
        
        Args:
            modality_idx: Which modality (0=vision, 1=audio, 2=proprio)
            seq_len: Number of tokens
            device: Target device
            
        Returns:
            Modality embedding [1, seq_len, dim]
        """
        idx = torch.tensor([modality_idx], device=device)
        emb = self.embeddings(idx)  # [1, dim]
        return emb.unsqueeze(1).expand(-1, seq_len, -1)  # [1, seq_len, dim]


class CrossModalAttentionLayer(nn.Module):
    """
    Single layer of cross-modal attention.
    
    Uses self-attention over concatenated modality tokens,
    allowing each token to attend to all other tokens
    regardless of modality.
    """
    
    def __init__(
        self, 
        dim: int, 
        num_heads: int,
        ff_dim: Optional[int] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        ff_dim = ff_dim or dim * 4
        
        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
            nn.Dropout(dropout),
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply cross-modal attention.
        
        Args:
            x: Input tokens [B, T, D]
            attn_mask: Optional attention mask [T, T]
            return_attention: Whether to return attention weights
            
        Returns:
            Tuple of:
            - Output tokens [B, T, D]
            - Attention weights [B, num_heads, T, T] (if return_attention=True)
        """
        # Self-attention with residual
        residual = x
        x = self.norm1(x)
        
        attn_out, attn_weights = self.self_attn(
            x, x, x,
            attn_mask=attn_mask,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        
        x = residual + self.dropout(attn_out)
        
        # Feed-forward with residual
        residual = x
        x = self.norm2(x)
        x = residual + self.ffn(x)
        
        if return_attention:
            return x, attn_weights
        return x, None


class CrossModalFusion(nn.Module):
    """
    Fuses multiple modalities using cross-attention.
    
    This module:
    1. Projects all modalities to a common dimension
    2. Adds modality embeddings to distinguish token types
    3. Concatenates all tokens
    4. Applies cross-modal attention layers
    
    The result is a unified representation where each token
    has attended to relevant information from all modalities.
    """
    
    def __init__(self, config: FusionConfig) -> None:
        super().__init__()
        
        self.config = config
        dim = config.dim
        
        # Modality embeddings
        self.modality_emb = ModalityEmbedding(dim)
        
        # Projection layers for each modality
        self.vision_proj = nn.Linear(dim, dim)
        self.audio_proj = nn.Linear(dim, dim)
        self.proprio_proj = nn.Linear(dim, dim)
        
        # Cross-modal attention layers
        self.layers = nn.ModuleList([
            CrossModalAttentionLayer(
                dim=dim,
                num_heads=config.num_heads,
                ff_dim=config.ff_dim,
                dropout=config.dropout,
            )
            for _ in range(config.num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(dim, dim)
        self.output_norm = nn.LayerNorm(dim)
    
    def forward(
        self,
        vision: torch.Tensor,
        audio: torch.Tensor,
        proprio: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Fuse multi-modal inputs.
        
        Args:
            vision: Vision features [B, T_v, D] or [B, H*W, D]
            audio: Audio features [B, T_a, D]
            proprio: Proprioception features [B, T_p, D]
            return_attention: Whether to return attention weights
            
        Returns:
            Tuple of:
            - Fused features [B, T_total, D]
            - List of attention weights from each layer (if return_attention)
        """
        B = vision.shape[0]
        device = vision.device
        
        # Project to common dimension
        v = self.vision_proj(vision)   # [B, T_v, D]
        a = self.audio_proj(audio)     # [B, T_a, D]
        p = self.proprio_proj(proprio) # [B, T_p, D]
        
        T_v, T_a, T_p = v.shape[1], a.shape[1], p.shape[1]
        
        # Add modality embeddings
        v = v + self.modality_emb(self.modality_emb.VISION, T_v, device)
        a = a + self.modality_emb(self.modality_emb.AUDIO, T_a, device)
        p = p + self.modality_emb(self.modality_emb.PROPRIO, T_p, device)
        
        # Concatenate all tokens
        all_tokens = torch.cat([v, a, p], dim=1)  # [B, T_v + T_a + T_p, D]
        
        # Apply cross-modal attention layers
        attn_weights_all = [] if return_attention else None
        
        for layer in self.layers:
            all_tokens, attn_weights = layer(
                all_tokens, 
                return_attention=return_attention
            )
            if return_attention:
                attn_weights_all.append(attn_weights)
        
        # Output projection
        fused = self.output_norm(all_tokens)
        fused = self.output_proj(fused)
        
        return fused, attn_weights_all
    
    def get_modality_slices(
        self, 
        T_v: int, 
        T_a: int, 
        T_p: int
    ) -> Dict[str, slice]:
        """
        Get slices for extracting modality-specific tokens from fused output.
        
        Returns:
            Dict mapping modality name to slice
        """
        return {
            'vision': slice(0, T_v),
            'audio': slice(T_v, T_v + T_a),
            'proprio': slice(T_v + T_a, T_v + T_a + T_p),
        }
    
    def extract_modality(
        self,
        fused: torch.Tensor,
        modality: str,
        T_v: int,
        T_a: int,
        T_p: int,
    ) -> torch.Tensor:
        """
        Extract tokens for a specific modality from fused output.
        
        Args:
            fused: Fused features [B, T_total, D]
            modality: 'vision', 'audio', or 'proprio'
            T_v, T_a, T_p: Token counts for each modality
            
        Returns:
            Modality-specific tokens [B, T_modality, D]
        """
        slices = self.get_modality_slices(T_v, T_a, T_p)
        return fused[:, slices[modality], :]


class GatedFusion(nn.Module):
    """
    Alternative fusion method using gating.
    
    Each modality contributes to the output weighted by
    learned gating values, allowing the model to dynamically
    weight modalities based on content.
    """
    
    def __init__(self, dim: int) -> None:
        super().__init__()
        
        # Gating networks for each modality
        self.vision_gate = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, 1),
            nn.Sigmoid(),
        )
        
        self.audio_gate = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, 1),
            nn.Sigmoid(),
        )
        
        self.proprio_gate = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, 1),
            nn.Sigmoid(),
        )
        
        # Output projection
        self.output_proj = nn.Linear(dim * 3, dim)
    
    def forward(
        self,
        vision: torch.Tensor,  # [B, D]
        audio: torch.Tensor,   # [B, D]
        proprio: torch.Tensor, # [B, D]
    ) -> torch.Tensor:
        """
        Fuse modalities using gating.
        
        Args:
            vision: Vision features [B, D]
            audio: Audio features [B, D]
            proprio: Proprioception features [B, D]
            
        Returns:
            Fused features [B, D]
        """
        # Compute gates
        g_v = self.vision_gate(vision)   # [B, 1]
        g_a = self.audio_gate(audio)     # [B, 1]
        g_p = self.proprio_gate(proprio) # [B, 1]
        
        # Weight and concatenate
        weighted = torch.cat([
            g_v * vision,
            g_a * audio,
            g_p * proprio,
        ], dim=-1)  # [B, D*3]
        
        return self.output_proj(weighted)  # [B, D]


class LateFusion(nn.Module):
    """
    Simple late fusion - concatenate and project.
    
    Useful as a baseline or when computational resources are limited.
    """
    
    def __init__(self, dim: int, num_modalities: int = 3) -> None:
        super().__init__()
        
        self.projection = nn.Sequential(
            nn.Linear(dim * num_modalities, dim * 2),
            nn.LayerNorm(dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )
    
    def forward(
        self,
        vision: torch.Tensor,  # [B, D]
        audio: torch.Tensor,   # [B, D]
        proprio: torch.Tensor, # [B, D]
    ) -> torch.Tensor:
        """
        Simple concatenation and projection.
        """
        concatenated = torch.cat([vision, audio, proprio], dim=-1)
        return self.projection(concatenated)
