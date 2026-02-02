"""
Cross-modal fusion using attention.

This module fuses multiple modalities (vision, audio, proprioception)
into unified representations using transformer-style attention.

The key insight: cross-modal attention learns which visual features
correspond to which sounds (e.g., a car horn -> the car in the image).

Enhanced Version (v2.0):
- Hierarchical fusion (early/mid/late)
- CLIP-style contrastive alignment
- Temporal synchronization for different sampling rates
- Cross-modal prediction heads
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FusionType(Enum):
    """Type of fusion architecture."""
    EARLY = "early"
    MID = "mid"
    LATE = "late"
    HIERARCHICAL = "hierarchical"


@dataclass
class FusionConfig:
    """Configuration for cross-modal fusion."""
    dim: int = 512
    num_heads: int = 8
    num_layers: int = 4
    dropout: float = 0.1
    ff_dim: Optional[int] = None  # Defaults to 4 * dim
    
    # Enhanced fusion settings
    fusion_type: str = "hierarchical"  # early, mid, late, hierarchical
    
    # Contrastive alignment
    use_contrastive: bool = True
    contrastive_temperature: float = 0.07
    contrastive_dim: int = 256  # Projection dimension for contrastive
    
    # Temporal synchronization
    use_temporal_sync: bool = True
    max_temporal_length: int = 100
    
    # Cross-modal prediction
    use_cross_modal_prediction: bool = True
    
    # Hierarchical fusion weights
    early_weight: float = 0.3
    mid_weight: float = 0.4
    late_weight: float = 0.3


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


# =============================================================================
# ENHANCED CROSS-MODAL FUSION (v2.0)
# =============================================================================

class ContrastiveAlignment(nn.Module):
    """
    CLIP-style contrastive alignment between modality pairs.
    
    Learns shared embedding spaces between:
    - Vision <-> Audio
    - Vision <-> Proprioception
    - Audio <-> Proprioception
    
    This encourages semantically related inputs across modalities
    to have similar representations.
    """
    
    def __init__(
        self,
        dim: int,
        projection_dim: int = 256,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        
        self.temperature = temperature
        self.projection_dim = projection_dim
        
        # Projection heads for each modality (following CLIP)
        self.vision_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, projection_dim),
        )
        
        self.audio_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, projection_dim),
        )
        
        self.proprio_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, projection_dim),
        )
        
        # Learnable temperature (like CLIP)
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(temperature)))
    
    def forward(
        self,
        vision: torch.Tensor,  # [B, D] or [B, T, D]
        audio: torch.Tensor,   # [B, D] or [B, T, D]
        proprio: torch.Tensor, # [B, D] or [B, T, D]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute contrastive embeddings and losses.
        
        Args:
            vision: Vision features
            audio: Audio features
            proprio: Proprioception features
            
        Returns:
            Dictionary with:
            - 'vision_emb': Vision projection [B, proj_dim]
            - 'audio_emb': Audio projection [B, proj_dim]
            - 'proprio_emb': Proprio projection [B, proj_dim]
            - 'loss_va': Vision-Audio contrastive loss
            - 'loss_vp': Vision-Proprio contrastive loss
            - 'loss_ap': Audio-Proprio contrastive loss
            - 'loss_total': Combined contrastive loss
        """
        # Pool temporal dimension if present
        if vision.dim() == 3:
            vision = vision.mean(dim=1)
        if audio.dim() == 3:
            audio = audio.mean(dim=1)
        if proprio.dim() == 3:
            proprio = proprio.mean(dim=1)
        
        # Project to shared space
        v_emb = F.normalize(self.vision_proj(vision), dim=-1)
        a_emb = F.normalize(self.audio_proj(audio), dim=-1)
        p_emb = F.normalize(self.proprio_proj(proprio), dim=-1)
        
        # Temperature
        temp = self.log_temperature.exp()
        
        # Compute pairwise contrastive losses
        loss_va = self._contrastive_loss(v_emb, a_emb, temp)
        loss_vp = self._contrastive_loss(v_emb, p_emb, temp)
        loss_ap = self._contrastive_loss(a_emb, p_emb, temp)
        
        loss_total = (loss_va + loss_vp + loss_ap) / 3
        
        return {
            'vision_emb': v_emb,
            'audio_emb': a_emb,
            'proprio_emb': p_emb,
            'loss_va': loss_va,
            'loss_vp': loss_vp,
            'loss_ap': loss_ap,
            'loss_total': loss_total,
        }
    
    def _contrastive_loss(
        self,
        emb1: torch.Tensor,
        emb2: torch.Tensor,
        temperature: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute symmetric InfoNCE loss between two sets of embeddings.
        """
        B = emb1.shape[0]
        
        # Similarity matrix
        logits = torch.matmul(emb1, emb2.T) / temperature
        
        # Labels: diagonal is positive pairs
        labels = torch.arange(B, device=emb1.device)
        
        # Symmetric loss
        loss_12 = F.cross_entropy(logits, labels)
        loss_21 = F.cross_entropy(logits.T, labels)
        
        return (loss_12 + loss_21) / 2
    
    def get_similarity(
        self,
        emb1: torch.Tensor,
        emb2: torch.Tensor,
    ) -> torch.Tensor:
        """Get similarity matrix between two sets of embeddings."""
        return torch.matmul(
            F.normalize(emb1, dim=-1),
            F.normalize(emb2, dim=-1).T
        )


class TemporalSynchronizer(nn.Module):
    """
    Learnable temporal alignment across modalities.
    
    Handles different sampling rates by learning to align
    temporal features across vision, audio, and proprioception.
    """
    
    def __init__(
        self,
        dim: int,
        max_length: int = 100,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        
        self.dim = dim
        self.max_length = max_length
        
        # Learnable temporal positional encodings
        self.temporal_pe = nn.Parameter(torch.randn(1, max_length, dim) * 0.02)
        
        # Cross-modal temporal attention
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True,
        )
        
        # Temporal interpolation network
        self.interp_net = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        
        # Output normalization
        self.norm = nn.LayerNorm(dim)
    
    def forward(
        self,
        features: torch.Tensor,  # [B, T, D]
        target_length: int,
        reference: Optional[torch.Tensor] = None,  # [B, T_ref, D]
    ) -> torch.Tensor:
        """
        Align features to target temporal length.
        
        Args:
            features: Input features [B, T, D]
            target_length: Desired output length
            reference: Optional reference features for alignment
            
        Returns:
            Aligned features [B, target_length, D]
        """
        B, T, D = features.shape
        
        # Add temporal positional encoding
        pe = self.temporal_pe[:, :T, :]
        features = features + pe
        
        # If reference provided, use cross-attention for alignment
        if reference is not None:
            T_ref = reference.shape[1]
            ref_pe = self.temporal_pe[:, :T_ref, :]
            reference = reference + ref_pe
            
            # Cross-attention: features attend to reference
            aligned, _ = self.temporal_attn(features, reference, reference)
            features = features + aligned
        
        # Interpolate to target length
        if T != target_length:
            # Use learned interpolation
            features = features.permute(0, 2, 1)  # [B, D, T]
            features = F.interpolate(
                features,
                size=target_length,
                mode='linear',
                align_corners=False
            )
            features = features.permute(0, 2, 1)  # [B, target_length, D]
        
        return self.norm(features)
    
    def synchronize_modalities(
        self,
        vision: torch.Tensor,   # [B, T_v, D]
        audio: torch.Tensor,    # [B, T_a, D]
        proprio: torch.Tensor,  # [B, T_p, D]
        target_length: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Synchronize all modalities to common temporal length.
        
        Args:
            vision: Vision features
            audio: Audio features
            proprio: Proprioception features
            target_length: Target length (defaults to max of inputs)
            
        Returns:
            Synchronized (vision, audio, proprio) all with same length
        """
        T_v, T_a, T_p = vision.shape[1], audio.shape[1], proprio.shape[1]
        
        if target_length is None:
            target_length = max(T_v, T_a, T_p)
        
        # Use vision as reference (usually highest resolution)
        vision_sync = self.forward(vision, target_length)
        audio_sync = self.forward(audio, target_length, reference=vision)
        proprio_sync = self.forward(proprio, target_length, reference=vision)
        
        return vision_sync, audio_sync, proprio_sync


class CrossModalPredictor(nn.Module):
    """
    Predicts one modality from another(s).
    
    Enables:
    - Vision -> Audio prediction (what does this look like it sounds like?)
    - Audio -> Vision prediction (what does this sound look like?)
    - Proprio -> Multimodal prediction (body state -> expected sensory input)
    """
    
    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        
        hidden_dim = hidden_dim or dim * 2
        
        # Vision to Audio predictor
        self.v2a = self._make_predictor(dim, hidden_dim, num_layers)
        
        # Audio to Vision predictor
        self.a2v = self._make_predictor(dim, hidden_dim, num_layers)
        
        # Vision to Proprio predictor
        self.v2p = self._make_predictor(dim, hidden_dim, num_layers)
        
        # Proprio to Vision predictor
        self.p2v = self._make_predictor(dim, hidden_dim, num_layers)
        
        # Audio to Proprio predictor
        self.a2p = self._make_predictor(dim, hidden_dim, num_layers)
        
        # Proprio to Audio predictor
        self.p2a = self._make_predictor(dim, hidden_dim, num_layers)
        
        # Multi-modal predictor (proprio -> V+A)
        self.p2va = self._make_predictor(dim, hidden_dim, num_layers, output_dim=dim * 2)
    
    def _make_predictor(
        self,
        dim: int,
        hidden_dim: int,
        num_layers: int,
        output_dim: Optional[int] = None,
    ) -> nn.Module:
        """Create a prediction MLP."""
        output_dim = output_dim or dim
        
        layers = []
        for i in range(num_layers):
            in_dim = dim if i == 0 else hidden_dim
            out_dim = output_dim if i == num_layers - 1 else hidden_dim
            
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
            ])
            
            if i < num_layers - 1:
                layers.append(nn.GELU())
        
        return nn.Sequential(*layers)
    
    def forward(
        self,
        source: torch.Tensor,
        source_modality: str,
        target_modality: str,
    ) -> torch.Tensor:
        """
        Predict target modality from source.
        
        Args:
            source: Source features [B, D] or [B, T, D]
            source_modality: 'vision', 'audio', or 'proprio'
            target_modality: 'vision', 'audio', 'proprio', or 'vision_audio'
            
        Returns:
            Predicted target features
        """
        # Pool if temporal
        if source.dim() == 3:
            source = source.mean(dim=1)
        
        # Select predictor
        pred_key = f"{source_modality[0]}2{target_modality[0]}"
        
        if pred_key == 'v2a':
            return self.v2a(source)
        elif pred_key == 'a2v':
            return self.a2v(source)
        elif pred_key == 'v2p':
            return self.v2p(source)
        elif pred_key == 'p2v':
            return self.p2v(source)
        elif pred_key == 'a2p':
            return self.a2p(source)
        elif pred_key == 'p2a':
            return self.p2a(source)
        elif source_modality == 'proprio' and target_modality == 'vision_audio':
            return self.p2va(source)
        else:
            raise ValueError(f"Unknown prediction: {source_modality} -> {target_modality}")
    
    def compute_prediction_loss(
        self,
        vision: torch.Tensor,
        audio: torch.Tensor,
        proprio: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all cross-modal prediction losses.
        
        Returns:
            Dictionary with individual and total losses
        """
        # Pool temporal dimension
        if vision.dim() == 3:
            vision = vision.mean(dim=1)
        if audio.dim() == 3:
            audio = audio.mean(dim=1)
        if proprio.dim() == 3:
            proprio = proprio.mean(dim=1)
        
        # Vision -> Audio
        audio_pred = self.v2a(vision)
        loss_v2a = F.mse_loss(audio_pred, audio)
        
        # Audio -> Vision
        vision_pred = self.a2v(audio)
        loss_a2v = F.mse_loss(vision_pred, vision)
        
        # Vision -> Proprio
        proprio_pred_v = self.v2p(vision)
        loss_v2p = F.mse_loss(proprio_pred_v, proprio)
        
        # Proprio -> Vision
        vision_pred_p = self.p2v(proprio)
        loss_p2v = F.mse_loss(vision_pred_p, vision)
        
        # Audio -> Proprio
        proprio_pred_a = self.a2p(audio)
        loss_a2p = F.mse_loss(proprio_pred_a, proprio)
        
        # Proprio -> Audio
        audio_pred_p = self.p2a(proprio)
        loss_p2a = F.mse_loss(audio_pred_p, audio)
        
        total_loss = (loss_v2a + loss_a2v + loss_v2p + loss_p2v + loss_a2p + loss_p2a) / 6
        
        return {
            'loss_v2a': loss_v2a,
            'loss_a2v': loss_a2v,
            'loss_v2p': loss_v2p,
            'loss_p2v': loss_p2v,
            'loss_a2p': loss_a2p,
            'loss_p2a': loss_p2a,
            'loss_total': total_loss,
        }


class EarlyFusion(nn.Module):
    """
    Early fusion: Concatenate and attend immediately.
    
    All modalities are combined at the input level before
    any processing, allowing maximum cross-modal interaction.
    """
    
    def __init__(self, config: FusionConfig) -> None:
        super().__init__()
        
        dim = config.dim
        
        # Initial projection
        self.proj = nn.Linear(dim * 3, dim)
        
        # Self-attention layers
        self.layers = nn.ModuleList([
            CrossModalAttentionLayer(
                dim=dim,
                num_heads=config.num_heads,
                ff_dim=config.ff_dim,
                dropout=config.dropout,
            )
            for _ in range(config.num_layers)
        ])
        
        self.norm = nn.LayerNorm(dim)
    
    def forward(
        self,
        vision: torch.Tensor,   # [B, D]
        audio: torch.Tensor,    # [B, D]
        proprio: torch.Tensor,  # [B, D]
    ) -> torch.Tensor:
        """Early fusion of pooled features."""
        # Concatenate
        combined = torch.cat([vision, audio, proprio], dim=-1)
        
        # Project
        x = self.proj(combined).unsqueeze(1)  # [B, 1, D]
        
        # Self-attention
        for layer in self.layers:
            x, _ = layer(x)
        
        return self.norm(x.squeeze(1))


class MidFusion(nn.Module):
    """
    Mid-level fusion using cross-attention.
    
    Each modality attends to the others, creating
    modality-specific representations enriched with
    cross-modal context.
    """
    
    def __init__(self, config: FusionConfig) -> None:
        super().__init__()
        
        dim = config.dim
        num_heads = config.num_heads
        
        # Cross-attention: vision attends to audio+proprio
        self.v_cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        
        # Cross-attention: audio attends to vision+proprio
        self.a_cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        
        # Cross-attention: proprio attends to vision+audio
        self.p_cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        
        # Layer norms
        self.norm_v = nn.LayerNorm(dim)
        self.norm_a = nn.LayerNorm(dim)
        self.norm_p = nn.LayerNorm(dim)
        
        # Final fusion
        self.fusion = nn.Linear(dim * 3, dim)
    
    def forward(
        self,
        vision: torch.Tensor,   # [B, T_v, D]
        audio: torch.Tensor,    # [B, T_a, D]
        proprio: torch.Tensor,  # [B, T_p, D]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Mid-level cross-attention fusion.
        
        Returns:
            (fused, vision_enriched, audio_enriched, proprio_enriched)
        """
        # Vision attends to audio + proprio
        ap = torch.cat([audio, proprio], dim=1)
        v_enriched, _ = self.v_cross_attn(vision, ap, ap)
        v_enriched = self.norm_v(vision + v_enriched)
        
        # Audio attends to vision + proprio
        vp = torch.cat([vision, proprio], dim=1)
        a_enriched, _ = self.a_cross_attn(audio, vp, vp)
        a_enriched = self.norm_a(audio + a_enriched)
        
        # Proprio attends to vision + audio
        va = torch.cat([vision, audio], dim=1)
        p_enriched, _ = self.p_cross_attn(proprio, va, va)
        p_enriched = self.norm_p(proprio + p_enriched)
        
        # Pool and fuse
        v_pooled = v_enriched.mean(dim=1)
        a_pooled = a_enriched.mean(dim=1)
        p_pooled = p_enriched.mean(dim=1)
        
        fused = self.fusion(torch.cat([v_pooled, a_pooled, p_pooled], dim=-1))
        
        return fused, v_enriched, a_enriched, p_enriched


class HierarchicalFusion(nn.Module):
    """
    Hierarchical fusion combining early, mid, and late fusion.
    
    This architecture:
    1. Early: Concatenates and processes together
    2. Mid: Cross-attention between modality pairs
    3. Late: Gated combination of processed features
    
    The outputs are weighted and combined for the final representation.
    """
    
    def __init__(self, config: FusionConfig) -> None:
        super().__init__()
        
        self.config = config
        dim = config.dim
        
        # Temporal synchronizer
        if config.use_temporal_sync:
            self.temporal_sync = TemporalSynchronizer(
                dim=dim,
                max_length=config.max_temporal_length,
            )
        else:
            self.temporal_sync = None
        
        # Early fusion path
        self.early_fusion = EarlyFusion(config)
        
        # Mid fusion path (cross-attention)
        self.mid_fusion = MidFusion(config)
        
        # Late fusion path
        self.late_fusion = LateFusion(dim)
        
        # Learnable fusion weights
        self.fusion_weights = nn.Parameter(torch.tensor([
            config.early_weight,
            config.mid_weight,
            config.late_weight,
        ]))
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
    
    def forward(
        self,
        vision: torch.Tensor,   # [B, T_v, D]
        audio: torch.Tensor,    # [B, T_a, D]
        proprio: torch.Tensor,  # [B, T_p, D]
        return_all_levels: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Hierarchical fusion of multi-modal inputs.
        
        Args:
            vision: Vision features
            audio: Audio features
            proprio: Proprioception features
            return_all_levels: Return outputs from all fusion levels
            
        Returns:
            Fused features [B, D] or dict with all level outputs
        """
        # Synchronize temporal dimensions if enabled
        if self.temporal_sync is not None:
            vision, audio, proprio = self.temporal_sync.synchronize_modalities(
                vision, audio, proprio
            )
        
        # Pool for early and late fusion
        v_pooled = vision.mean(dim=1)
        a_pooled = audio.mean(dim=1)
        p_pooled = proprio.mean(dim=1)
        
        # Early fusion
        early_out = self.early_fusion(v_pooled, a_pooled, p_pooled)
        
        # Mid fusion (uses full temporal features)
        mid_out, v_enriched, a_enriched, p_enriched = self.mid_fusion(
            vision, audio, proprio
        )
        
        # Late fusion
        late_out = self.late_fusion(v_pooled, a_pooled, p_pooled)
        
        # Weighted combination
        weights = F.softmax(self.fusion_weights, dim=0)
        fused = (
            weights[0] * early_out +
            weights[1] * mid_out +
            weights[2] * late_out
        )
        
        fused = self.output_proj(fused)
        
        if return_all_levels:
            return {
                'fused': fused,
                'early': early_out,
                'mid': mid_out,
                'late': late_out,
                'vision_enriched': v_enriched,
                'audio_enriched': a_enriched,
                'proprio_enriched': p_enriched,
                'fusion_weights': weights,
            }
        
        return fused


class EnhancedCrossModalFusion(nn.Module):
    """
    Enhanced cross-modal fusion with all advanced features.
    
    Combines:
    - Hierarchical fusion (early/mid/late)
    - CLIP-style contrastive alignment
    - Temporal synchronization
    - Cross-modal prediction heads
    
    This is the main fusion module for NSCA v2.0.
    """
    
    def __init__(self, config: FusionConfig) -> None:
        super().__init__()
        
        self.config = config
        dim = config.dim
        
        # Modality projections (same as original)
        self.vision_proj = nn.Linear(dim, dim)
        self.audio_proj = nn.Linear(dim, dim)
        self.proprio_proj = nn.Linear(dim, dim)
        
        # Modality embeddings
        self.modality_emb = ModalityEmbedding(dim)
        
        # Hierarchical fusion
        self.hierarchical_fusion = HierarchicalFusion(config)
        
        # Contrastive alignment (optional)
        if config.use_contrastive:
            self.contrastive = ContrastiveAlignment(
                dim=dim,
                projection_dim=config.contrastive_dim,
                temperature=config.contrastive_temperature,
            )
        else:
            self.contrastive = None
        
        # Cross-modal prediction (optional)
        if config.use_cross_modal_prediction:
            self.predictor = CrossModalPredictor(dim=dim)
        else:
            self.predictor = None
        
        # Original attention-based fusion (for backward compatibility)
        self.legacy_fusion = CrossModalFusion(config)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
    
    def forward(
        self,
        vision: torch.Tensor,
        audio: torch.Tensor,
        proprio: torch.Tensor,
        return_losses: bool = False,
        return_attention: bool = False,
        use_legacy: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Enhanced cross-modal fusion.
        
        Args:
            vision: Vision features [B, T_v, D]
            audio: Audio features [B, T_a, D]
            proprio: Proprioception features [B, T_p, D]
            return_losses: Return contrastive and prediction losses
            return_attention: Return attention weights (legacy mode only)
            use_legacy: Use original attention-based fusion
            
        Returns:
            Dictionary with:
            - 'fused': Main fused output [B, D]
            - 'fused_sequence': Sequence output [B, T, D] (if use_legacy)
            - 'hierarchical': Hierarchical fusion outputs (dict)
            - 'contrastive': Contrastive alignment outputs (if return_losses)
            - 'prediction': Prediction losses (if return_losses)
        """
        B = vision.shape[0]
        device = vision.device
        
        # Project modalities
        v = self.vision_proj(vision)
        a = self.audio_proj(audio)
        p = self.proprio_proj(proprio)
        
        results = {}
        
        # Legacy mode for backward compatibility
        if use_legacy:
            fused_seq, attn_weights = self.legacy_fusion(
                v, a, p, return_attention=return_attention
            )
            results['fused_sequence'] = fused_seq
            results['fused'] = fused_seq.mean(dim=1)
            
            if return_attention:
                results['attention_weights'] = attn_weights
            
            return results
        
        # Hierarchical fusion
        hierarchical_out = self.hierarchical_fusion(
            v, a, p, return_all_levels=True
        )
        results['fused'] = hierarchical_out['fused']
        results['hierarchical'] = hierarchical_out
        
        # Contrastive alignment
        if return_losses and self.contrastive is not None:
            contrastive_out = self.contrastive(v, a, p)
            results['contrastive'] = contrastive_out
        
        # Cross-modal prediction
        if return_losses and self.predictor is not None:
            prediction_out = self.predictor.compute_prediction_loss(v, a, p)
            results['prediction'] = prediction_out
        
        return results
    
    def compute_loss(
        self,
        vision: torch.Tensor,
        audio: torch.Tensor,
        proprio: torch.Tensor,
        contrastive_weight: float = 0.3,
        prediction_weight: float = 0.25,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all fusion-related losses.
        
        Args:
            vision, audio, proprio: Input features
            contrastive_weight: Weight for contrastive loss
            prediction_weight: Weight for prediction loss
            
        Returns:
            Dictionary with individual and total losses
        """
        results = self.forward(
            vision, audio, proprio,
            return_losses=True,
        )
        
        losses = {}
        total_loss = torch.tensor(0.0, device=vision.device)
        
        if 'contrastive' in results:
            losses['contrastive_loss'] = results['contrastive']['loss_total']
            total_loss = total_loss + contrastive_weight * losses['contrastive_loss']
        
        if 'prediction' in results:
            losses['prediction_loss'] = results['prediction']['loss_total']
            total_loss = total_loss + prediction_weight * losses['prediction_loss']
        
        losses['total_loss'] = total_loss
        
        return losses
    
    def get_modality_embeddings(
        self,
        vision: torch.Tensor,
        audio: torch.Tensor,
        proprio: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Get contrastive embeddings for each modality.
        
        Useful for retrieval tasks (find images that match sounds, etc.)
        """
        if self.contrastive is None:
            raise ValueError("Contrastive alignment not enabled")
        
        v = self.vision_proj(vision)
        a = self.audio_proj(audio)
        p = self.proprio_proj(proprio)
        
        result = self.contrastive(v, a, p)
        
        return {
            'vision': result['vision_emb'],
            'audio': result['audio_emb'],
            'proprio': result['proprio_emb'],
        }


# Factory function for creating fusion modules
def create_fusion_module(
    config: Optional[FusionConfig] = None,
    fusion_type: str = "hierarchical",
    dim: int = 512,
    use_contrastive: bool = True,
    use_prediction: bool = True,
) -> nn.Module:
    """
    Factory function to create fusion modules.
    
    Args:
        config: Full configuration (overrides other args)
        fusion_type: Type of fusion ('early', 'mid', 'late', 'hierarchical', 'legacy')
        dim: Feature dimension
        use_contrastive: Enable contrastive alignment
        use_prediction: Enable cross-modal prediction
        
    Returns:
        Fusion module
    """
    if config is None:
        config = FusionConfig(
            dim=dim,
            fusion_type=fusion_type,
            use_contrastive=use_contrastive,
            use_cross_modal_prediction=use_prediction,
        )
    
    if fusion_type == "legacy":
        return CrossModalFusion(config)
    else:
        return EnhancedCrossModalFusion(config)
