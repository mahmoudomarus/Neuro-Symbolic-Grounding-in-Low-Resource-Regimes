"""
Counterfactual Reasoning - "What if...?" thinking.

Enables reasoning about alternative possibilities:
- "What would have happened if I hadn't pushed it?"
- "What would happen if this were made of glass instead of steel?"
- "Would the ball still have fallen if there was no gravity?"

This is essential for:
- Learning from mistakes
- Planning alternative actions
- Understanding causation deeply
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CounterfactualQuery:
    """A counterfactual question."""
    actual_state: torch.Tensor
    hypothetical_change: str  # Description of what's different
    variable_indices: List[int]  # Which state dimensions to modify
    modification: torch.Tensor  # How to modify them


@dataclass
class CounterfactualResult:
    """Result of counterfactual reasoning."""
    predicted_outcome: torch.Tensor
    confidence: float
    causal_attribution: Dict[str, float]  # Which factors mattered
    explanation: str


class CounterfactualReasoner(nn.Module):
    """
    Reason about counterfactual scenarios.
    
    "What would have happened if X were different?"
    
    This requires:
    1. Identifying causal structure
    2. Intervening on specific variables
    3. Propagating effects through causal graph
    """
    
    def __init__(
        self,
        state_dim: int = 256,
        hidden_dim: int = 512,
        num_factors: int = 16,
    ) -> None:
        super().__init__()
        
        self.state_dim = state_dim
        self.num_factors = num_factors
        
        # Factor encoder (disentangle state into causal factors)
        self.factor_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_factors),
        )
        
        # Factor decoder (reconstruct state from factors)
        self.factor_decoder = nn.Sequential(
            nn.Linear(num_factors, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, state_dim),
        )
        
        # Causal mechanism (how factors influence each other)
        self.causal_mechanism = nn.Sequential(
            nn.Linear(num_factors, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_factors),
        )
        
        # Confidence estimator
        self.confidence_head = nn.Sequential(
            nn.Linear(state_dim + num_factors, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
    
    def encode_to_factors(self, state: torch.Tensor) -> torch.Tensor:
        """Encode state into disentangled causal factors."""
        return self.factor_encoder(state)
    
    def decode_from_factors(self, factors: torch.Tensor) -> torch.Tensor:
        """Decode factors back to state."""
        return self.factor_decoder(factors)
    
    def intervene(
        self,
        state: torch.Tensor,
        factor_idx: int,
        new_value: float,
    ) -> torch.Tensor:
        """
        Perform intervention: set a factor to specific value.
        
        This is the key operation for counterfactual reasoning:
        "What if factor X were set to Y?"
        """
        # Encode to factors
        factors = self.encode_to_factors(state)
        
        # Intervene
        factors_intervened = factors.clone()
        factors_intervened[..., factor_idx] = new_value
        
        # Propagate through causal mechanism
        factors_propagated = self.causal_mechanism(factors_intervened)
        
        # Decode back
        counterfactual_state = self.decode_from_factors(factors_propagated)
        
        return counterfactual_state
    
    def what_if(
        self,
        actual_state: torch.Tensor,
        hypothetical_factors: torch.Tensor,
    ) -> CounterfactualResult:
        """
        Answer "What would happen if factors were different?"
        
        Args:
            actual_state: The actual observed state
            hypothetical_factors: Modified factor values
            
        Returns:
            CounterfactualResult with predicted outcome
        """
        # Propagate hypothetical factors
        propagated = self.causal_mechanism(hypothetical_factors)
        
        # Decode to state
        predicted_outcome = self.decode_from_factors(propagated)
        
        # Estimate confidence
        confidence_input = torch.cat([actual_state, hypothetical_factors], dim=-1)
        confidence = self.confidence_head(confidence_input).squeeze(-1).mean().item()
        
        # Compute causal attribution (which factors mattered most)
        actual_factors = self.encode_to_factors(actual_state)
        factor_diff = (hypothetical_factors - actual_factors).abs()
        
        if factor_diff.dim() > 1:
            factor_diff = factor_diff[0]
        
        causal_attribution = {
            f"factor_{i}": factor_diff[i].item()
            for i in range(self.num_factors)
        }
        
        # Generate explanation
        most_changed = max(causal_attribution, key=causal_attribution.get)
        explanation = f"The counterfactual scenario differs mainly in {most_changed}"
        
        return CounterfactualResult(
            predicted_outcome=predicted_outcome,
            confidence=confidence,
            causal_attribution=causal_attribution,
            explanation=explanation,
        )
    
    def would_outcome_change(
        self,
        state_before: torch.Tensor,
        state_after: torch.Tensor,
        action: torch.Tensor,
        hypothetical_action: torch.Tensor,
    ) -> Tuple[bool, float, torch.Tensor]:
        """
        Would the outcome be different with a different action?
        
        This is key for learning: "If I had done something else,
        would things have turned out better?"
        
        Returns:
            - would_change: bool
            - change_magnitude: float
            - hypothetical_outcome: tensor
        """
        # Encode action effect
        actual_effect = state_after - state_before
        
        # Predict what would happen with different action
        # (Simplified - assumes action effect is roughly linear)
        action_diff = hypothetical_action - action
        predicted_effect_change = action_diff.sum(dim=-1, keepdim=True) * 0.1
        
        hypothetical_outcome = state_before + actual_effect + predicted_effect_change
        
        # Measure change
        change_magnitude = (hypothetical_outcome - state_after).norm(dim=-1).mean().item()
        would_change = change_magnitude > 0.1
        
        return would_change, change_magnitude, hypothetical_outcome
    
    def causal_responsibility(
        self,
        state_before: torch.Tensor,
        state_after: torch.Tensor,
        candidate_causes: List[torch.Tensor],
    ) -> List[float]:
        """
        Determine how much each candidate cause contributed to the effect.
        
        Uses counterfactual criterion: A caused B if B wouldn't have
        happened without A.
        """
        actual_factors = self.encode_to_factors(state_after)
        
        responsibilities = []
        
        for cause in candidate_causes:
            # Imagine removing this cause
            factors_without = self.encode_to_factors(state_before)
            
            # Would effect still happen?
            propagated = self.causal_mechanism(factors_without)
            hypothetical_after = self.decode_from_factors(propagated)
            
            # Difference = how much this cause contributed
            contribution = (state_after - hypothetical_after).norm(dim=-1).mean().item()
            responsibilities.append(contribution)
        
        # Normalize
        total = sum(responsibilities) + 1e-8
        responsibilities = [r / total for r in responsibilities]
        
        return responsibilities
