"""
Attention Allocation - What to focus on.

Determines where to direct limited cognitive resources:
- Novel stimuli attract attention
- Dangerous stimuli attract attention
- Goal-relevant stimuli attract attention
- Predictable stimuli can be ignored

This is closely tied to drives: we attend to what's relevant
to our current needs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .drive_system import DriveState


@dataclass
class AttentionTarget:
    """Something that could receive attention."""
    location: Tuple[float, float]  # Spatial location
    salience: float                # How attention-grabbing
    relevance: float              # How relevant to goals
    novelty: float                # How novel
    threat_level: float           # How dangerous


class SalienceComputer(nn.Module):
    """
    Compute bottom-up salience of stimuli.
    
    Salience is automatic attention-grabbing:
    - Bright things are salient
    - Moving things are salient
    - Loud things are salient
    - Sudden changes are salient
    """
    
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        
        self.salience_net = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Compute salience from features."""
        return self.salience_net(features).squeeze(-1)


class RelevanceComputer(nn.Module):
    """
    Compute top-down relevance to current goals.
    
    Relevance depends on current state:
    - If hungry, food is relevant
    - If curious, novel things are relevant
    - If scared, escape routes are relevant
    """
    
    def __init__(self, feature_dim: int, drive_dim: int = 5) -> None:
        super().__init__()
        
        self.relevance_net = nn.Sequential(
            nn.Linear(feature_dim + drive_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
    
    def forward(
        self,
        features: torch.Tensor,
        drive_state: DriveState,
    ) -> torch.Tensor:
        """Compute relevance given current drives."""
        drive_tensor = drive_state.to_tensor().to(features.device)
        
        if features.dim() > 1:
            drive_tensor = drive_tensor.unsqueeze(0).expand(features.shape[0], -1)
        
        combined = torch.cat([features, drive_tensor], dim=-1)
        return self.relevance_net(combined).squeeze(-1)


class NoveltyDetector(nn.Module):
    """
    Detect novelty of stimuli.
    
    Novel things automatically attract attention.
    This is innate - babies look longer at novel stimuli.
    """
    
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        
        # Running statistics of seen features
        self.register_buffer('feature_mean', torch.zeros(feature_dim))
        self.register_buffer('feature_var', torch.ones(feature_dim))
        self.count = 0
    
    def update(self, features: torch.Tensor) -> None:
        """Update running statistics."""
        if features.dim() > 1:
            features = features.mean(dim=0)
        
        self.count += 1
        delta = features - self.feature_mean
        self.feature_mean = self.feature_mean + delta / self.count
        self.feature_var = self.feature_var + delta * (features - self.feature_mean)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Compute novelty score."""
        if self.count < 10:
            return torch.ones(features.shape[0], device=features.device)
        
        var = self.feature_var / max(1, self.count - 1) + 1e-6
        distance = ((features - self.feature_mean) ** 2 / var).sum(dim=-1)
        
        # Normalize to [0, 1]
        novelty = 1 - torch.exp(-distance / features.shape[-1])
        
        return novelty


class ThreatDetector(nn.Module):
    """
    Detect threats that should attract attention.
    
    Threats get priority attention (for survival).
    This is partially innate - babies show fear of heights,
    snakes, spiders even without experience.
    """
    
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        
        self.threat_net = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Estimate threat level."""
        return self.threat_net(features).squeeze(-1)


class AttentionAllocator(nn.Module):
    """
    Allocate attention across stimuli.
    
    Combines:
    - Bottom-up salience (automatic)
    - Top-down relevance (goal-directed)
    - Novelty (curiosity-driven)
    - Threat detection (survival-driven)
    """
    
    def __init__(self, feature_dim: int, drive_dim: int = 5) -> None:
        super().__init__()
        
        self.salience = SalienceComputer(feature_dim)
        self.relevance = RelevanceComputer(feature_dim, drive_dim)
        self.novelty = NoveltyDetector(feature_dim)
        self.threat = ThreatDetector(feature_dim)
        
        # Attention weights
        self.attention_weights = nn.Parameter(torch.tensor([
            0.3,  # salience
            0.3,  # relevance
            0.2,  # novelty
            0.2,  # threat
        ]))
    
    def forward(
        self,
        features: torch.Tensor,
        drive_state: DriveState,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute attention allocation.
        
        Args:
            features: Features of stimuli [B, N, D] or [B, D]
            drive_state: Current drive state
            
        Returns:
            - Attention weights [B, N] or [B]
            - Component breakdown
        """
        if features.dim() == 2:
            # Single stimulus per batch
            features = features.unsqueeze(1)
            squeeze_output = True
        else:
            squeeze_output = False
        
        B, N, D = features.shape
        
        # Compute components for each stimulus
        features_flat = features.reshape(B * N, D)
        
        salience_scores = self.salience(features_flat).reshape(B, N)
        
        # Relevance needs drive state
        relevance_scores = []
        for i in range(N):
            rel = self.relevance(features[:, i, :], drive_state)
            relevance_scores.append(rel)
        relevance_scores = torch.stack(relevance_scores, dim=1)
        
        novelty_scores = self.novelty(features_flat).reshape(B, N)
        threat_scores = self.threat(features_flat).reshape(B, N)
        
        # Combine with learned weights
        weights = F.softmax(self.attention_weights, dim=0)
        
        attention = (
            weights[0] * salience_scores +
            weights[1] * relevance_scores +
            weights[2] * novelty_scores +
            weights[3] * threat_scores
        )
        
        # Normalize across stimuli
        attention = F.softmax(attention, dim=-1)
        
        # Update novelty detector
        self.novelty.update(features_flat.mean(dim=0))
        
        components = {
            'salience': salience_scores,
            'relevance': relevance_scores,
            'novelty': novelty_scores,
            'threat': threat_scores,
        }
        
        if squeeze_output:
            attention = attention.squeeze(1)
            components = {k: v.squeeze(1) for k, v in components.items()}
        
        return attention, components
    
    def focus_on_threat(self, threat_level: float) -> None:
        """Bias attention toward threats when danger detected."""
        if threat_level > 0.7:
            self.attention_weights.data[3] += 0.1  # Increase threat weight
            self.attention_weights.data[2] -= 0.05  # Decrease novelty
    
    def focus_on_novelty(self) -> None:
        """Bias attention toward novelty (curiosity mode)."""
        self.attention_weights.data[2] += 0.1
        self.attention_weights.data[0] -= 0.05
