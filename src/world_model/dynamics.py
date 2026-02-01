"""
Dynamics predictor (Imagination): (z_t, action) -> z_{t+1}.

Fuses the current latent map with an action by broadcasting the action across
the spatial grid, then processes with ResBlocks to predict the next latent map.
Spatial [H, W] dimensions are preserved throughout.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .config import DynamicsConfig
from .encoder import ResBlock


class DynamicsPredictor(nn.Module):
    """
    Predicts next latent state z_{t+1} from current latent map z_t and action.

    Fuses z_t [B, C, H, W] with action [B, A_dim] by broadcasting action to
    [B, A_dim, H, W] and concatenating, then runs a stack of ResBlocks to produce
    a predicted latent map [B, C, H, W]. Spatial dimensions are never flattened.
    """

    def __init__(self, config: DynamicsConfig) -> None:
        super().__init__()
        self.config = config
        latent_dim = config.latent_dim
        action_dim = config.action_dim
        hidden_dim = config.hidden_dim
        num_layers = config.num_layers

        fused_channels = latent_dim + action_dim
        self.fuse_proj = nn.Sequential(
            nn.Conv2d(fused_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        blocks: list[nn.Module] = []
        for _ in range(num_layers):
            blocks.append(ResBlock(hidden_dim, hidden_dim, stride=1))
        self.blocks = nn.Sequential(*blocks)

        self.out_proj = nn.Conv2d(hidden_dim, latent_dim, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Predict next latent map from current latent map and action.

        Args:
            z: Current latent map [B, C, H, W].
            action: Action vector [B, A_dim].

        Returns:
            Predicted next latent map [B, C, H, W], same spatial shape as z.
        """
        B, C, H, W = z.shape
        if C != self.config.latent_dim:
            raise ValueError(
                f"z channels {C} does not match config.latent_dim {self.config.latent_dim}."
            )
        if action.shape[0] != B or action.shape[1] != self.config.action_dim:
            raise ValueError(
                f"action shape {action.shape} incompatible with B={B}, action_dim={self.config.action_dim}."
            )

        # Broadcast action to [B, A_dim, H, W] and concatenate with z
        action_spatial = action.view(B, -1, 1, 1).expand(B, -1, H, W)
        fused = torch.cat([z, action_spatial], dim=1)

        x = self.fuse_proj(fused)
        x = self.blocks(x)
        return self.out_proj(x)
