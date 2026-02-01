"""
Enhanced Dynamics Predictor.

Predicts future world states from current state + action.
This is the "imagination engine" - it allows the agent to
mentally simulate what would happen if it took certain actions.

Key features:
- Multi-step trajectory prediction
- Uncertainty estimation (know what you don't know)
- Residual prediction (predict change, not absolute state)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class EnhancedDynamicsConfig:
    """Configuration for enhanced dynamics predictor."""
    state_dim: int = 256
    action_dim: int = 32
    hidden_dim: int = 512
    num_layers: int = 3
    dropout: float = 0.1
    use_residual: bool = True
    predict_uncertainty: bool = True


class EnhancedDynamicsPredictor(nn.Module):
    """
    Predicts future world states from current state + action.
    
    This is the core of the world model's ability to "imagine"
    consequences of actions before taking them.
    
    Features:
    - Residual prediction (predicts delta, not absolute)
    - Uncertainty estimation
    - Multi-step trajectory prediction
    """
    
    def __init__(self, config: EnhancedDynamicsConfig) -> None:
        super().__init__()
        
        self.config = config
        
        # State + Action fusion
        self.fusion = nn.Sequential(
            nn.Linear(config.state_dim + config.action_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        
        # Prediction network (residual blocks)
        self.predictor = nn.ModuleList([
            PredictorBlock(config.hidden_dim, config.dropout)
            for _ in range(config.num_layers)
        ])
        
        # Output projection for state delta
        self.state_proj = nn.Linear(config.hidden_dim, config.state_dim)
        
        # Uncertainty head (predicts how confident the model is)
        if config.predict_uncertainty:
            self.uncertainty_head = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim // 4),
                nn.GELU(),
                nn.Linear(config.hidden_dim // 4, 1),
                nn.Sigmoid(),
            )
        else:
            self.uncertainty_head = None
    
    def forward(
        self, 
        state: torch.Tensor, 
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Predict next state from current state and action.
        
        Args:
            state: Current world state [B, state_dim]
            action: Action to take [B, action_dim]
            
        Returns:
            Tuple of:
            - Predicted next state [B, state_dim]
            - Uncertainty estimate [B, 1] (if enabled)
        """
        # Fuse state and action
        fused = self.fusion(torch.cat([state, action], dim=-1))  # [B, hidden_dim]
        
        # Predict through residual blocks
        hidden = fused
        for block in self.predictor:
            hidden = block(hidden)
        
        # Predict state delta
        delta = self.state_proj(hidden)  # [B, state_dim]
        
        # Residual prediction: next_state = current_state + delta
        if self.config.use_residual:
            next_state = state + delta
        else:
            next_state = delta
        
        # Uncertainty estimation
        if self.uncertainty_head is not None:
            uncertainty = self.uncertainty_head(hidden)  # [B, 1]
        else:
            uncertainty = None
        
        return next_state, uncertainty
    
    def predict_trajectory(
        self, 
        initial_state: torch.Tensor, 
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Predict multiple steps into the future.
        
        Args:
            initial_state: Starting world state [B, state_dim]
            actions: Sequence of actions [B, T, action_dim]
            
        Returns:
            Tuple of:
            - Predicted states [B, T+1, state_dim] (includes initial)
            - Uncertainties [B, T, 1] (if enabled)
        """
        B, T, _ = actions.shape
        
        states = [initial_state]
        uncertainties = []
        
        state = initial_state
        for t in range(T):
            action = actions[:, t, :]
            state, uncertainty = self.forward(state, action)
            states.append(state)
            if uncertainty is not None:
                uncertainties.append(uncertainty)
        
        states = torch.stack(states, dim=1)  # [B, T+1, state_dim]
        
        if uncertainties:
            uncertainties = torch.stack(uncertainties, dim=1)  # [B, T, 1]
        else:
            uncertainties = None
        
        return states, uncertainties
    
    def imagine_outcomes(
        self,
        state: torch.Tensor,
        action_candidates: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Imagine outcomes for multiple action candidates.
        
        Useful for planning: given current state, what would happen
        if we took each of these possible actions?
        
        Args:
            state: Current state [B, state_dim]
            action_candidates: Candidate actions [B, K, action_dim]
            
        Returns:
            Tuple of:
            - Predicted next states [B, K, state_dim]
            - Uncertainties [B, K, 1]
        """
        B, K, A = action_candidates.shape
        
        # Expand state for each candidate
        state_expanded = state.unsqueeze(1).expand(-1, K, -1)  # [B, K, state_dim]
        
        # Flatten for batch processing
        state_flat = state_expanded.reshape(B * K, -1)
        action_flat = action_candidates.reshape(B * K, -1)
        
        # Predict
        next_state_flat, uncertainty_flat = self.forward(state_flat, action_flat)
        
        # Reshape back
        next_states = next_state_flat.reshape(B, K, -1)
        
        if uncertainty_flat is not None:
            uncertainties = uncertainty_flat.reshape(B, K, -1)
        else:
            uncertainties = None
        
        return next_states, uncertainties


class PredictorBlock(nn.Module):
    """Residual block for prediction network."""
    
    def __init__(self, dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        
        self.activation = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.net(x))


class LatentDynamicsPredictor(nn.Module):
    """
    Dynamics predictor that operates on spatial latent maps.
    
    Unlike the vector-based predictor, this preserves spatial structure
    and can predict how visual features change over time.
    """
    
    def __init__(
        self,
        latent_channels: int = 64,
        action_dim: int = 32,
        hidden_channels: int = 128,
    ) -> None:
        super().__init__()
        
        self.latent_channels = latent_channels
        self.action_dim = action_dim
        
        # Action embedding network
        self.action_embed = nn.Sequential(
            nn.Linear(action_dim, latent_channels * 2),
            nn.GELU(),
            nn.Linear(latent_channels * 2, latent_channels),
        )
        
        # Prediction network (spatial)
        fused_channels = latent_channels + latent_channels  # state + action
        self.predictor = nn.Sequential(
            nn.Conv2d(fused_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, latent_channels, kernel_size=3, padding=1),
        )
    
    def forward(
        self, 
        z: torch.Tensor, 
        action: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict next latent map from current latent map and action.
        
        Args:
            z: Current latent map [B, C, H, W]
            action: Action vector [B, action_dim]
            
        Returns:
            Predicted next latent map [B, C, H, W]
        """
        B, C, H, W = z.shape
        
        # Embed action and broadcast to spatial dimensions
        action_emb = self.action_embed(action)  # [B, C]
        action_spatial = action_emb.view(B, C, 1, 1).expand(-1, -1, H, W)  # [B, C, H, W]
        
        # Fuse state and action
        fused = torch.cat([z, action_spatial], dim=1)  # [B, 2C, H, W]
        
        # Predict delta
        delta = self.predictor(fused)  # [B, C, H, W]
        
        # Residual
        z_next = z + delta
        
        return z_next


class WorldModelPredictor(nn.Module):
    """
    Full world model predictor combining multiple prediction heads.
    
    Can predict:
    - Next world state
    - Uncertainty
    - Reward (optional)
    - Termination (optional)
    """
    
    def __init__(self, config: EnhancedDynamicsConfig) -> None:
        super().__init__()
        
        self.config = config
        
        # Core dynamics predictor
        self.dynamics = EnhancedDynamicsPredictor(config)
        
        # Optional reward head
        self.reward_head = nn.Sequential(
            nn.Linear(config.state_dim, config.hidden_dim // 4),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 4, 1),
        )
        
        # Optional termination head (done prediction)
        self.done_head = nn.Sequential(
            nn.Linear(config.state_dim, config.hidden_dim // 4),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 4, 1),
            nn.Sigmoid(),
        )
    
    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> dict:
        """
        Predict next state, reward, and termination.
        
        Returns dict with keys: next_state, uncertainty, reward, done
        """
        next_state, uncertainty = self.dynamics(state, action)
        
        return {
            'next_state': next_state,
            'uncertainty': uncertainty,
            'reward': self.reward_head(next_state),
            'done': self.done_head(next_state),
        }
