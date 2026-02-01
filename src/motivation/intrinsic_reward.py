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
from typing import Any, Dict, List, Optional, Tuple

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
    Compute reward from curiosity satisfaction (LEGACY - use RobustCuriosityReward).
    
    Based on the idea that prediction error is inherently rewarding
    (up to a point) because it indicates learning opportunity.
    
    WARNING: This basic version is vulnerable to the "noisy TV" problem
    where random noise produces high but useless prediction error.
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


class RobustCuriosityReward(nn.Module):
    """
    Curiosity reward with "noisy TV" defense.
    
    The Problem (identified in peer review):
    Pure prediction error rewards random noise (the "noisy TV" problem).
    A TV showing static would have high prediction error forever,
    but there's nothing to learn from it.
    
    The Solution:
    Reward = prediction_error * learnability
    
    Where learnability = how much error DECREASES with exposure.
    - Noisy TV: High error, but no decrease → low learnability → low reward
    - Novel physics: High error, decreases with learning → high reward
    
    Implementation:
    1. EMA encoder for stable state hashing (Savinov et al., 2018)
    2. Track error history per state hash
    3. Compute learnability as error reduction
    
    References:
    - Burda et al. (2018). Large-Scale Study of Curiosity-Driven Learning.
    - Savinov et al. (2018). Episodic Curiosity through Reachability.
    """
    
    def __init__(
        self,
        state_dim: int,
        hash_dim: int = 64,
        ema_decay: float = 0.99,
        history_window: int = 100,
        min_history_for_learnability: int = 5,
    ) -> None:
        super().__init__()
        
        self.state_dim = state_dim
        self.hash_dim = hash_dim
        self.ema_decay = ema_decay
        self.history_window = history_window
        self.min_history = min_history_for_learnability
        
        # Forward model for prediction error
        self.forward_model = nn.Sequential(
            nn.Linear(state_dim * 2 + 32, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, state_dim),
        )
        
        # Fast encoder (updated by gradients)
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.GELU(),
            nn.Linear(128, hash_dim),
        )
        
        # Slow encoder (EMA of fast encoder, for stable hashing)
        self.slow_encoder = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.GELU(),
            nn.Linear(128, hash_dim),
        )
        # Initialize slow encoder with same weights
        for slow_p, fast_p in zip(self.slow_encoder.parameters(), self.encoder.parameters()):
            slow_p.data.copy_(fast_p.data)
            slow_p.requires_grad = False
        
        # Track prediction error history per state hash
        self.error_history: Dict[Tuple[int, ...], List[float]] = {}
        
    def compute_state_hash(self, state: torch.Tensor) -> Tuple[int, ...]:
        """
        Compute locality-sensitive hash for state.
        
        Uses slow encoder for stability - the hash shouldn't change
        rapidly as the fast encoder learns.
        """
        with torch.no_grad():
            embedding = self.slow_encoder(state)
            # Binarize to create hash (LSH-style)
            hash_bits = (embedding > 0).int()
            
            # Convert to tuple for dict key
            if hash_bits.dim() > 1:
                hash_bits = hash_bits[0]  # Take first in batch
            
            return tuple(hash_bits.cpu().tolist())
    
    def update_slow_encoder(self) -> None:
        """
        EMA update of slow encoder.
        
        slow_params = decay * slow_params + (1 - decay) * fast_params
        """
        with torch.no_grad():
            for slow_p, fast_p in zip(
                self.slow_encoder.parameters(), 
                self.encoder.parameters()
            ):
                slow_p.data = (
                    self.ema_decay * slow_p.data + 
                    (1 - self.ema_decay) * fast_p.data
                )
    
    def compute_learnability(self, state_hash: Tuple[int, ...]) -> float:
        """
        Compute learnability for a state.
        
        Learnability = (early_error - recent_error) / early_error
        
        High learnability: Error decreases (we're learning something)
        Low learnability: Error stays high (might be random/unlearnable)
        """
        history = self.error_history.get(state_hash, [])
        
        if len(history) < self.min_history:
            return 1.0  # Assume learnable until proven otherwise
        
        window = min(len(history) // 2, 20)
        if window < 3:
            return 1.0
        
        early_error = sum(history[:window]) / window
        recent_error = sum(history[-window:]) / window
        
        if early_error < 1e-6:
            return 0.0  # Already at zero error
        
        learnability = (early_error - recent_error) / (early_error + 1e-6)
        return max(0.0, learnability)  # Clamp to non-negative
    
    def forward(
        self,
        state: torch.Tensor,
        next_state: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute robust curiosity reward.
        
        reward = prediction_error * learnability
        
        Returns:
            - reward: Curiosity reward tensor
            - diagnostics: Dict with prediction_error, learnability, etc.
        """
        # Compute prediction error
        forward_input = torch.cat([state, action], dim=-1)
        if forward_input.shape[-1] < state.shape[-1] * 2 + 32:
            pad_size = state.shape[-1] * 2 + 32 - forward_input.shape[-1]
            forward_input = F.pad(forward_input, (0, pad_size))
        
        predicted_next = self.forward_model(forward_input)
        prediction_error = (predicted_next - next_state).pow(2).mean(dim=-1)
        
        # Compute state hash and learnability
        state_hash = self.compute_state_hash(state)
        learnability = self.compute_learnability(state_hash)
        
        # Update error history
        error_value = prediction_error.mean().item()
        if state_hash not in self.error_history:
            self.error_history[state_hash] = []
        self.error_history[state_hash].append(error_value)
        
        # Bound history size
        if len(self.error_history[state_hash]) > self.history_window:
            self.error_history[state_hash] = self.error_history[state_hash][-self.history_window:]
        
        # Compute reward
        # High error + high learnability = high reward (worth learning)
        # High error + low learnability = low reward (noisy TV)
        reward = prediction_error.clamp(0, 1) * learnability
        
        # Update slow encoder (EMA)
        self.update_slow_encoder()
        
        diagnostics = {
            'prediction_error': prediction_error.mean().item(),
            'learnability': learnability,
            'raw_reward': prediction_error.mean().item(),
            'filtered_reward': reward.mean().item(),
            'history_length': len(self.error_history.get(state_hash, [])),
            'unique_states_seen': len(self.error_history),
        }
        
        return reward, diagnostics
    
    def get_curiosity_statistics(self) -> Dict[str, Any]:
        """Get statistics about curiosity state."""
        if not self.error_history:
            return {'unique_states': 0, 'avg_history_length': 0}
        
        history_lengths = [len(h) for h in self.error_history.values()]
        
        return {
            'unique_states': len(self.error_history),
            'avg_history_length': sum(history_lengths) / len(history_lengths),
            'max_history_length': max(history_lengths),
            'total_observations': sum(history_lengths),
        }
    
    def reset_history(self) -> None:
        """Reset error history (for new episode/task)."""
        self.error_history.clear()


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
    - Curiosity (prediction error with noisy TV defense)
    - Competence (mastery)
    - Information gain (uncertainty reduction)
    
    ARCHITECTURAL UPDATE (Peer Review):
    Now uses RobustCuriosityReward by default to defend against
    the "noisy TV" problem.
    """
    
    def __init__(
        self,
        state_dim: int,
        curiosity_weight: float = 0.4,
        competence_weight: float = 0.4,
        info_gain_weight: float = 0.2,
        use_robust_curiosity: bool = True,  # NEW: use robust version by default
    ) -> None:
        super().__init__()
        
        self.curiosity_weight = curiosity_weight
        self.competence_weight = competence_weight
        self.info_gain_weight = info_gain_weight
        self.use_robust_curiosity = use_robust_curiosity
        
        # Use robust curiosity by default (noisy TV defense)
        if use_robust_curiosity:
            self.curiosity = RobustCuriosityReward(state_dim)
        else:
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
        # Compute curiosity reward
        if self.use_robust_curiosity:
            curiosity_r, curiosity_diagnostics = self.curiosity(state, next_state, action)
            pred_error = curiosity_diagnostics.get('prediction_error', 0.0)
        else:
            curiosity_r, pred_error = self.curiosity(state, next_state, action)
        
        # Compute other rewards
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
    
    def get_curiosity_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostics from curiosity module."""
        if self.use_robust_curiosity and hasattr(self.curiosity, 'get_curiosity_statistics'):
            return self.curiosity.get_curiosity_statistics()
        return {}
