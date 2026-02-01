"""
Intrinsic Reward Computation.

Computes reward signals from internal drives rather than
external feedback:
- Curiosity reward: Novelty and learning opportunity
- Competence reward: Successful prediction and control
- Information gain: Reduction in uncertainty

This is what makes agents learn without external supervision.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RewardComponents:
    """Breakdown of intrinsic reward."""
    curiosity: float
    competence: float
    information_gain: float
    total: float
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'curiosity': self.curiosity,
            'competence': self.competence,
            'information_gain': self.information_gain,
            'total': self.total,
        }


class CuriosityReward(nn.Module):
    """
    Compute reward from curiosity satisfaction.
    
    Based on the idea that prediction error is inherently rewarding
    (up to a point) because it indicates learning opportunity.
    """
    
    def __init__(self, state_dim: int) -> None:
        super().__init__()
        
        # ICM-style curiosity (Intrinsic Curiosity Module)
        # Forward model: predicts next state from current state + action
        self.forward_model = nn.Sequential(
            nn.Linear(state_dim * 2 + 32, 256),  # state + action
            nn.GELU(),
            nn.Linear(256, state_dim),
        )
        
        # Inverse model: predicts action from states
        self.inverse_model = nn.Sequential(
            nn.Linear(state_dim * 2, 256),
            nn.GELU(),
            nn.Linear(256, 32),
        )
    
    def forward(
        self,
        state: torch.Tensor,
        next_state: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute curiosity reward.
        
        Reward = prediction error of forward model
        
        The idea: If we can't predict what happens, it's interesting.
        """
        # Forward model prediction
        forward_input = torch.cat([state, action], dim=-1)
        # Pad if needed
        if forward_input.shape[-1] < state.shape[-1] * 2 + 32:
            pad_size = state.shape[-1] * 2 + 32 - forward_input.shape[-1]
            forward_input = F.pad(forward_input, (0, pad_size))
        
        predicted_next = self.forward_model(forward_input)
        
        # Prediction error = curiosity reward
        prediction_error = (predicted_next - next_state).pow(2).mean(dim=-1)
        
        # Scale to reasonable range
        curiosity_reward = prediction_error.clamp(0, 1)
        
        return curiosity_reward, prediction_error


class CompetenceReward(nn.Module):
    """
    Compute reward from competence/mastery.
    
    Reward for:
    - Successfully predicting outcomes
    - Achieving intended effects
    - Reducing uncertainty
    """
    
    def __init__(self, state_dim: int) -> None:
        super().__init__()
        
        # Track prediction history
        self.prediction_history: list = []
        self.max_history = 50
        
        # Competence estimator
        self.competence_net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
    
    def record_prediction(
        self,
        predicted: torch.Tensor,
        actual: torch.Tensor,
    ) -> float:
        """Record a prediction outcome."""
        error = (predicted - actual).norm(dim=-1).mean().item()
        accuracy = max(0, 1 - error)
        
        self.prediction_history.append(accuracy)
        if len(self.prediction_history) > self.max_history:
            self.prediction_history.pop(0)
        
        return accuracy
    
    def get_learning_progress(self) -> float:
        """How much has prediction accuracy improved?"""
        if len(self.prediction_history) < 10:
            return 0.0
        
        early = sum(self.prediction_history[:10]) / 10
        recent = sum(self.prediction_history[-10:]) / 10
        
        return max(0, recent - early)
    
    def forward(
        self,
        state: torch.Tensor,
        prediction_accuracy: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Compute competence reward.
        
        Reward for:
        - High prediction accuracy
        - Learning progress (improvement)
        """
        # Base competence from network
        base_competence = self.competence_net(state).squeeze(-1)
        
        if prediction_accuracy is not None:
            # Blend with actual accuracy
            reward = 0.5 * base_competence + 0.5 * prediction_accuracy
        else:
            reward = base_competence
        
        # Bonus for learning progress
        progress = self.get_learning_progress()
        reward = reward + 0.2 * progress
        
        return reward


class InformationGainReward(nn.Module):
    """
    Reward for reducing uncertainty (information gain).
    
    The agent is rewarded for actions that reduce its uncertainty
    about the world. This is a Bayesian view of curiosity.
    """
    
    def __init__(self, state_dim: int) -> None:
        super().__init__()
        
        # Uncertainty estimator
        self.uncertainty_net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.GELU(),
            nn.Linear(128, state_dim),  # Predict variance per dimension
            nn.Softplus(),
        )
    
    def forward(
        self,
        state_before: torch.Tensor,
        state_after: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute information gain reward.
        
        Reward = reduction in uncertainty
        """
        uncertainty_before = self.uncertainty_net(state_before)
        uncertainty_after = self.uncertainty_net(state_after)
        
        # Information gain = reduction in uncertainty
        info_gain = (uncertainty_before - uncertainty_after).sum(dim=-1)
        
        # Positive reward for reducing uncertainty
        reward = F.relu(info_gain)
        
        return reward


class IntrinsicRewardComputer(nn.Module):
    """
    Complete intrinsic reward computation.
    
    Combines multiple reward signals:
    - Curiosity (prediction error)
    - Competence (mastery)
    - Information gain (uncertainty reduction)
    """
    
    def __init__(
        self,
        state_dim: int,
        curiosity_weight: float = 0.4,
        competence_weight: float = 0.4,
        info_gain_weight: float = 0.2,
    ) -> None:
        super().__init__()
        
        self.curiosity_weight = curiosity_weight
        self.competence_weight = competence_weight
        self.info_gain_weight = info_gain_weight
        
        self.curiosity = CuriosityReward(state_dim)
        self.competence = CompetenceReward(state_dim)
        self.info_gain = InformationGainReward(state_dim)
    
    def forward(
        self,
        state: torch.Tensor,
        next_state: torch.Tensor,
        action: torch.Tensor,
        prediction_accuracy: Optional[float] = None,
    ) -> Tuple[torch.Tensor, RewardComponents]:
        """
        Compute total intrinsic reward.
        
        Returns:
            - Total reward tensor
            - Breakdown of components
        """
        # Individual rewards
        curiosity_r, pred_error = self.curiosity(state, next_state, action)
        competence_r = self.competence(state, prediction_accuracy)
        info_r = self.info_gain(state, next_state)
        
        # Combine
        total = (
            self.curiosity_weight * curiosity_r +
            self.competence_weight * competence_r +
            self.info_gain_weight * info_r
        )
        
        # Create breakdown
        components = RewardComponents(
            curiosity=curiosity_r.mean().item(),
            competence=competence_r.mean().item(),
            information_gain=info_r.mean().item(),
            total=total.mean().item(),
        )
        
        return total, components
