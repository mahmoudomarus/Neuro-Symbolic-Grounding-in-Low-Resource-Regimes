"""
Drive System - Innate motivations that create learning pressure.

Humans have innate drives that motivate behavior:
- Curiosity: Seek novel, learnable experiences
- Competence: Seek mastery over environment
- Social: Seek connection, attention, imitation
- Homeostatic: Maintain energy, avoid pain

Without these drives, there's no reason to learn.
A baby learns to breastfeed because hunger creates urgency.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F


class DriveType(Enum):
    """Types of innate drives."""
    CURIOSITY = "curiosity"         # Seek novelty and learning
    COMPETENCE = "competence"       # Seek mastery
    SOCIAL = "social"               # Seek connection
    ENERGY = "energy"               # Maintain energy (simulated)
    SAFETY = "safety"               # Avoid danger


@dataclass
class DriveState:
    """Current state of all drives."""
    curiosity_level: float = 0.5    # 0 = satisfied, 1 = urgent
    competence_level: float = 0.5
    social_level: float = 0.5
    energy_level: float = 0.8       # 0 = exhausted, 1 = full
    safety_level: float = 0.9       # 0 = danger, 1 = safe
    
    def to_tensor(self) -> torch.Tensor:
        """Convert to tensor."""
        return torch.tensor([
            self.curiosity_level,
            self.competence_level,
            self.social_level,
            self.energy_level,
            self.safety_level,
        ])
    
    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> 'DriveState':
        """Create from tensor."""
        return cls(
            curiosity_level=tensor[0].item(),
            competence_level=tensor[1].item(),
            social_level=tensor[2].item(),
            energy_level=tensor[3].item(),
            safety_level=tensor[4].item(),
        )
    
    def most_urgent(self) -> DriveType:
        """Get the most urgent drive."""
        drives = {
            DriveType.CURIOSITY: self.curiosity_level,
            DriveType.COMPETENCE: self.competence_level,
            DriveType.SOCIAL: self.social_level,
            DriveType.ENERGY: 1 - self.energy_level,  # Invert (low energy = urgent)
            DriveType.SAFETY: 1 - self.safety_level,   # Invert
        }
        return max(drives, key=drives.get)


@dataclass
class DriveConfig:
    """Configuration for drive system."""
    state_dim: int = 256
    hidden_dim: int = 128
    num_drives: int = 5
    curiosity_decay: float = 0.01     # How fast curiosity decays
    competence_growth: float = 0.1    # How fast competence grows with success
    energy_decay: float = 0.001       # How fast energy depletes


class CuriosityDrive(nn.Module):
    """
    Curiosity drive - seek novel, learnable experiences.
    
    Curiosity is not random exploration. It's targeted:
    - Seek things that are novel (not seen before)
    - But also learnable (not too complex)
    - "Goldilocks zone" of complexity
    
    Based on prediction error:
    - High prediction error = interesting
    - But too high = overwhelming
    """
    
    def __init__(self, state_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        
        # Novelty detector
        self.novelty_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        
        # Learnability estimator (is this comprehensible?)
        self.learnability_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        
        # Memory of seen states (for novelty)
        self.register_buffer(
            'seen_states_mean',
            torch.zeros(1, state_dim)
        )
        self.register_buffer(
            'seen_states_var',
            torch.ones(1, state_dim)
        )
        self.seen_count = 0
    
    def update_memory(self, state: torch.Tensor) -> None:
        """Update running statistics of seen states."""
        if state.dim() > 1:
            state = state.mean(dim=0, keepdim=True)
        
        self.seen_count += 1
        
        # Running mean and variance
        delta = state - self.seen_states_mean
        self.seen_states_mean = self.seen_states_mean + delta / self.seen_count
        self.seen_states_var = self.seen_states_var + delta * (state - self.seen_states_mean)
    
    def compute_novelty(self, state: torch.Tensor) -> torch.Tensor:
        """How novel is this state compared to what we've seen?"""
        if self.seen_count < 10:
            # Everything is novel at first
            return torch.ones(state.shape[0], device=state.device)
        
        # Mahalanobis-like distance from seen distribution
        diff = state - self.seen_states_mean
        var = self.seen_states_var / max(1, self.seen_count - 1) + 1e-6
        distance = (diff ** 2 / var).sum(dim=-1)
        
        # Normalize to [0, 1]
        novelty = 1 - torch.exp(-distance / state.shape[-1])
        
        return novelty
    
    def compute_learnability(self, state: torch.Tensor) -> torch.Tensor:
        """Is this state learnable (not too complex)?"""
        return self.learnability_net(state).squeeze(-1)
    
    def forward(
        self,
        state: torch.Tensor,
        prediction_error: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute curiosity signal.
        
        High curiosity for:
        - Novel states (not seen before)
        - Learnable states (not too complex)
        - High prediction error (surprising)
        """
        novelty = self.compute_novelty(state)
        learnability = self.compute_learnability(state)
        
        if prediction_error is not None:
            # Prediction error is a strong curiosity signal
            pred_err_normalized = torch.sigmoid(prediction_error)
            curiosity = 0.3 * novelty + 0.3 * learnability + 0.4 * pred_err_normalized
        else:
            curiosity = 0.5 * novelty + 0.5 * learnability
        
        return curiosity


class CompetenceDrive(nn.Module):
    """
    Competence drive - seek mastery over environment.
    
    Humans want to be good at things:
    - Successfully predicting outcomes = satisfying
    - Controlling environment = satisfying
    - Failing repeatedly = frustrating
    
    This creates pressure to improve skills.
    """
    
    def __init__(self, state_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        
        # Track prediction accuracy history
        self.accuracy_history: List[float] = []
        self.max_history = 100
        
        # Competence estimator
        self.competence_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
    
    def record_outcome(self, predicted: torch.Tensor, actual: torch.Tensor) -> float:
        """Record prediction outcome for competence tracking."""
        error = (predicted - actual).norm(dim=-1).mean().item()
        accuracy = max(0, 1 - error)
        
        self.accuracy_history.append(accuracy)
        if len(self.accuracy_history) > self.max_history:
            self.accuracy_history.pop(0)
        
        return accuracy
    
    def get_recent_competence(self) -> float:
        """Get recent competence level."""
        if not self.accuracy_history:
            return 0.5
        return sum(self.accuracy_history[-20:]) / len(self.accuracy_history[-20:])
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute competence satisfaction.
        
        High when:
        - Recent predictions were accurate
        - Feel in control of environment
        """
        estimated_competence = self.competence_net(state).squeeze(-1)
        recent_competence = self.get_recent_competence()
        
        # Blend estimated with actual
        return 0.5 * estimated_competence + 0.5 * recent_competence


class DriveSystem(nn.Module):
    """
    Complete drive system with multiple innate drives.
    
    Creates learning pressure through:
    - Curiosity (seek novelty)
    - Competence (seek mastery)
    - Prediction error minimization
    
    Without these, the agent has no reason to learn.
    """
    
    def __init__(self, config: DriveConfig) -> None:
        super().__init__()
        
        self.config = config
        
        # Individual drives
        self.curiosity = CuriosityDrive(config.state_dim, config.hidden_dim)
        self.competence = CompetenceDrive(config.state_dim, config.hidden_dim)
        
        # Drive integration
        self.drive_integrator = nn.Sequential(
            nn.Linear(config.num_drives, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, 1),
        )
        
        # Current drive state
        self.drive_state = DriveState()
        
        # Energy dynamics
        self.energy_decay = config.energy_decay
    
    def forward(
        self,
        world_state: torch.Tensor,
        prediction_error: Optional[torch.Tensor] = None,
    ) -> Tuple[DriveState, torch.Tensor]:
        """
        Compute current drive state and motivation signal.
        
        Args:
            world_state: Current world state
            prediction_error: How wrong was our prediction?
            
        Returns:
            - Updated DriveState
            - Overall motivation signal [B]
        """
        # Compute individual drives
        curiosity_signal = self.curiosity(world_state, prediction_error)
        competence_signal = self.competence(world_state)
        
        # Update curiosity memory
        self.curiosity.update_memory(world_state)
        
        # Update drive state
        self.drive_state.curiosity_level = curiosity_signal.mean().item()
        self.drive_state.competence_level = competence_signal.mean().item()
        
        # Energy decay
        self.drive_state.energy_level = max(
            0.1,
            self.drive_state.energy_level - self.energy_decay
        )
        
        # Compute overall motivation
        drive_tensor = self.drive_state.to_tensor().to(world_state.device)
        drive_tensor = drive_tensor.unsqueeze(0).expand(world_state.shape[0], -1)
        
        motivation = self.drive_integrator(drive_tensor).squeeze(-1)
        
        return self.drive_state, motivation
    
    def get_intrinsic_reward(
        self,
        state: torch.Tensor,
        prediction_error: torch.Tensor,
        learning_progress: float,
    ) -> torch.Tensor:
        """
        Compute intrinsic reward from drives.
        
        This is what makes the agent WANT to learn.
        """
        # Curiosity reward: High prediction error is interesting
        curiosity_reward = torch.sigmoid(prediction_error - 0.5)
        
        # Competence reward: Learning progress is satisfying
        competence_reward = torch.full_like(curiosity_reward, learning_progress)
        
        # Safety penalty: Avoid dangerous states
        safety_penalty = 0  # Would need danger detector
        
        # Combine
        reward = (
            0.4 * curiosity_reward +
            0.4 * competence_reward +
            0.2 * self.drive_state.energy_level  # Maintain energy
        ) - safety_penalty
        
        return reward
    
    def should_explore(self) -> bool:
        """Should the agent explore (vs exploit)?"""
        return self.drive_state.curiosity_level > 0.6
    
    def should_rest(self) -> bool:
        """Should the agent rest (low energy)?"""
        return self.drive_state.energy_level < 0.2
    
    def recover_energy(self, amount: float = 0.1) -> None:
        """Recover energy (e.g., after rest)."""
        self.drive_state.energy_level = min(
            1.0,
            self.drive_state.energy_level + amount
        )
