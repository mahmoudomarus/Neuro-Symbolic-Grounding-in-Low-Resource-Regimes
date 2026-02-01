"""
Affordance Detection - What can be done with objects.

Affordances are action possibilities that objects offer:
- A chair affords sitting
- A ball affords throwing/rolling
- A cup affords containing/drinking

Key insight: Affordances are perceived directly from properties,
not learned from language labels. A flat, stable surface "affords"
sitting even if you've never seen a chair.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .property_layer import PropertyVector


@dataclass
class AffordanceConfig:
    """Configuration for affordance detection."""
    property_dim: int = 9  # From PropertyVector
    hidden_dim: int = 128
    num_affordances: int = 12


class AffordanceVector:
    """
    Vector of affordance scores.
    
    Each affordance is a probability that the action is possible.
    """
    
    AFFORDANCE_NAMES = [
        "graspable",      # Can be picked up
        "sittable",       # Can sit on it
        "containable",    # Can hold things
        "throwable",      # Can be thrown
        "rollable",       # Can roll
        "stackable",      # Can stack on/under
        "breakable",      # Can be broken
        "edible",         # Can be eaten (context-dependent)
        "wearable",       # Can be worn
        "openable",       # Can be opened
        "pushable",       # Can be pushed/moved
        "climbable",      # Can be climbed
    ]
    
    def __init__(self, scores: torch.Tensor):
        """
        Args:
            scores: Affordance scores [B, num_affordances]
        """
        self.scores = scores
    
    def get(self, affordance: str) -> torch.Tensor:
        """Get score for specific affordance."""
        idx = self.AFFORDANCE_NAMES.index(affordance)
        return self.scores[..., idx]
    
    def top_affordances(self, k: int = 3) -> List[Tuple[str, float]]:
        """Get top k affordances."""
        if self.scores.dim() > 1:
            scores = self.scores[0]  # Take first in batch
        else:
            scores = self.scores
        
        top_k = torch.topk(scores, k)
        return [
            (self.AFFORDANCE_NAMES[idx], scores[idx].item())
            for idx in top_k.indices
        ]


class AffordanceDetector(nn.Module):
    """
    Detect affordances from semantic properties.
    
    Maps property vectors to affordance possibilities.
    Some mappings are straightforward (graspable = small + not too heavy),
    others require learning from experience.
    """
    
    def __init__(self, config: AffordanceConfig) -> None:
        super().__init__()
        
        self.config = config
        num_affordances = len(AffordanceVector.AFFORDANCE_NAMES)
        
        # Affordance prediction network
        self.network = nn.Sequential(
            nn.Linear(config.property_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, num_affordances),
            nn.Sigmoid(),
        )
        
        # Innate affordance priors (rules that don't need learning)
        # These encode obvious relationships
        self.register_buffer('graspable_weights', torch.tensor([
            0.0,   # hardness - doesn't matter
            -0.5,  # weight - lighter is more graspable
            -0.8,  # size - smaller is more graspable
            0.0,   # animacy - doesn't matter
            0.0,   # rigidity
            0.0,   # transparency
            0.0,   # roughness
            0.0,   # temperature
            0.0,   # containment
        ]))
        
        self.register_buffer('sittable_weights', torch.tensor([
            0.3,   # hardness - somewhat hard is better
            0.0,   # weight - doesn't matter
            0.5,   # size - needs to be big enough
            -1.0,  # animacy - don't sit on living things
            0.5,   # rigidity - rigid is better
            0.0,   # transparency
            0.0,   # roughness
            0.0,   # temperature
            0.0,   # containment
        ]))
    
    def forward(
        self,
        properties: PropertyVector,
        use_priors: bool = True,
    ) -> AffordanceVector:
        """
        Detect affordances from properties.
        
        Args:
            properties: Semantic property vector
            use_priors: Whether to use innate affordance priors
            
        Returns:
            AffordanceVector with scores for each affordance
        """
        prop_tensor = properties.to_tensor()
        
        # Learned affordance detection
        learned_scores = self.network(prop_tensor)
        
        if use_priors:
            # Add innate priors for basic affordances
            graspable_prior = torch.sigmoid(
                (prop_tensor * self.graspable_weights).sum(dim=-1, keepdim=True) + 0.5
            )
            sittable_prior = torch.sigmoid(
                (prop_tensor * self.sittable_weights).sum(dim=-1, keepdim=True)
            )
            
            # Blend learned with priors
            scores = learned_scores.clone()
            scores[..., 0] = 0.5 * scores[..., 0] + 0.5 * graspable_prior.squeeze(-1)
            scores[..., 1] = 0.5 * scores[..., 1] + 0.5 * sittable_prior.squeeze(-1)
        else:
            scores = learned_scores
        
        return AffordanceVector(scores)
    
    def explain_affordance(
        self,
        affordance: str,
        properties: PropertyVector,
    ) -> str:
        """
        Explain why an affordance is/isn't present.
        
        Useful for debugging and language grounding.
        """
        prop_tensor = properties.to_tensor()
        if prop_tensor.dim() > 1:
            prop_tensor = prop_tensor[0]
        
        explanations = []
        
        if affordance == "graspable":
            size = prop_tensor[2].item()
            weight = prop_tensor[1].item()
            if size > 0.7:
                explanations.append("too large to grasp")
            if weight > 0.8:
                explanations.append("too heavy to lift")
            if not explanations:
                explanations.append("small and light enough to grasp")
        
        elif affordance == "sittable":
            size = prop_tensor[2].item()
            rigidity = prop_tensor[4].item()
            animacy = prop_tensor[3].item()
            if size < 0.3:
                explanations.append("too small to sit on")
            if rigidity < 0.3:
                explanations.append("too soft/flexible")
            if animacy > 0.5:
                explanations.append("it's alive - shouldn't sit on it")
            if not explanations:
                explanations.append("stable and large enough to sit on")
        
        return "; ".join(explanations) if explanations else "uncertain"
