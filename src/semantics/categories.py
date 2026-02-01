"""
Category Classification - Fundamental object categories.

Categorizes objects into fundamental types:
- Animate vs Inanimate
- Agent vs Object
- Natural vs Artificial
- Living vs Non-living

These categories are partially innate - babies distinguish animate
from inanimate motion from birth.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F

from .property_layer import PropertyVector


class FundamentalCategory(Enum):
    """Fundamental ontological categories."""
    AGENT = "agent"           # Self-propelled, goal-directed
    OBJECT = "object"         # Passive, no self-motion
    SUBSTANCE = "substance"   # Continuous (water, sand)
    CONTAINER = "container"   # Can hold other things
    SURFACE = "surface"       # Flat, can support things
    TOOL = "tool"            # Used to achieve goals


@dataclass
class CategoryScores:
    """Scores for each fundamental category."""
    agent: float
    object: float
    substance: float
    container: float
    surface: float
    tool: float
    
    def primary_category(self) -> FundamentalCategory:
        """Get the most likely category."""
        scores = {
            FundamentalCategory.AGENT: self.agent,
            FundamentalCategory.OBJECT: self.object,
            FundamentalCategory.SUBSTANCE: self.substance,
            FundamentalCategory.CONTAINER: self.container,
            FundamentalCategory.SURFACE: self.surface,
            FundamentalCategory.TOOL: self.tool,
        }
        return max(scores, key=scores.get)


class CategoryClassifier(nn.Module):
    """
    Classify objects into fundamental categories.
    
    Uses property vectors to determine category:
    - Agent: High animacy, self-propelled motion
    - Object: Bounded, movable, inanimate
    - Substance: No fixed shape, flows
    - Container: Has interior space, can hold things
    - Surface: Large, flat, supports things
    - Tool: Small, graspable, used for manipulation
    """
    
    def __init__(self, property_dim: int = 9, hidden_dim: int = 64) -> None:
        super().__init__()
        
        self.num_categories = len(FundamentalCategory)
        
        # Category classification network
        self.classifier = nn.Sequential(
            nn.Linear(property_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.num_categories),
        )
        
        # Innate category priors (partially hardcoded)
        # Agent detection is partially innate
        self.register_buffer('agent_weights', torch.tensor([
            0.0,   # hardness
            0.0,   # weight
            0.0,   # size
            2.0,   # animacy - PRIMARY indicator
            0.0,   # rigidity
            0.0,   # transparency
            0.0,   # roughness
            0.0,   # temperature
            0.0,   # containment
        ]))
        
        # Container detection
        self.register_buffer('container_weights', torch.tensor([
            0.3,   # hardness - somewhat hard
            0.0,   # weight
            0.3,   # size - medium size
            -1.0,  # animacy - not animate
            0.5,   # rigidity
            0.0,   # transparency
            0.0,   # roughness
            0.0,   # temperature
            2.0,   # containment - PRIMARY
        ]))
    
    def forward(self, properties: PropertyVector) -> Tuple[torch.Tensor, CategoryScores]:
        """
        Classify into fundamental categories.
        
        Args:
            properties: Property vector from Layer 1
            
        Returns:
            Tuple of:
            - Category logits [B, num_categories]
            - CategoryScores for first item in batch
        """
        prop_tensor = properties.to_tensor()
        
        # Learned classification
        logits = self.classifier(prop_tensor)
        
        # Add innate priors for agent and container
        agent_prior = (prop_tensor * self.agent_weights).sum(dim=-1, keepdim=True)
        container_prior = (prop_tensor * self.container_weights).sum(dim=-1, keepdim=True)
        
        # Blend with learned
        logits[..., 0] = logits[..., 0] + agent_prior.squeeze(-1)  # Agent
        logits[..., 3] = logits[..., 3] + container_prior.squeeze(-1)  # Container
        
        # Convert to scores
        probs = F.softmax(logits, dim=-1)
        
        # Create CategoryScores for first item
        if probs.dim() > 1:
            p = probs[0]
        else:
            p = probs
        
        scores = CategoryScores(
            agent=p[0].item(),
            object=p[1].item(),
            substance=p[2].item(),
            container=p[3].item(),
            surface=p[4].item(),
            tool=p[5].item(),
        )
        
        return logits, scores
    
    def is_agent(self, properties: PropertyVector, threshold: float = 0.5) -> torch.Tensor:
        """Quick check if something is an agent."""
        return properties.animacy > threshold
    
    def is_graspable_object(self, properties: PropertyVector) -> torch.Tensor:
        """Check if it's a graspable object (small, not agent)."""
        is_small = properties.size < 0.5
        is_not_agent = properties.animacy < 0.3
        is_not_heavy = properties.weight < 0.7
        return is_small & is_not_agent & is_not_heavy


class AnimateInanimateClassifier(nn.Module):
    """
    Simple binary classifier for animate vs inanimate.
    
    This is a fundamental distinction that humans make from birth.
    Infants prefer looking at biological motion over random motion.
    """
    
    def __init__(self, feature_dim: int = 256) -> None:
        super().__init__()
        
        # Very simple - animacy is mostly about motion patterns
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Classify as animate (1) or inanimate (0).
        
        Args:
            features: Visual/motion features [B, D]
            
        Returns:
            Animacy probability [B]
        """
        return self.classifier(features).squeeze(-1)
