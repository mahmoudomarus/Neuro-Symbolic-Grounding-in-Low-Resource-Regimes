"""
JEPA learning algorithm: predicts next latent state from current state + action.

Uses a Siamese setup (same encoder for current and next frame). Loss is latent
consistency (MSE or cosine) plus variance regularization (VICReg-style) to prevent
collapse.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DynamicsConfig, EncoderConfig, GeometryConfig, JEPAConfig
from .dynamics import DynamicsPredictor
from .encoder import SpatialEncoder


def latent_consistency_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """MSE between predicted and target latent maps (same shape [B, C, H, W])."""
    return F.mse_loss(prediction, target)


def variance_regularization(
    prediction: torch.Tensor,
    threshold: float = 1.0,
) -> torch.Tensor:
    """
    Penalize when std of prediction (over batch) is below threshold (anti-collapse).
    Std is computed per (c, h, w), then we penalize mean(relu(threshold - std)).
    """
    # prediction [B, C, H, W]; std over dim=0 -> [C, H, W]
    std = prediction.std(dim=0)
    return F.relu(threshold - std).mean()


class JEPA(nn.Module):
    """
    Joint-Embedding Predictive Architecture: encoder + dynamics predictor.

    Encodes current frame x_t (with grad) and next frame x_{t+1} (target, no grad).
    Predicts z_{t+1} from z_t and action. Loss = consistency (MSE) + variance
    regularization to prevent collapse.
    """

    def __init__(
        self,
        encoder: SpatialEncoder,
        predictor: DynamicsPredictor,
        jepa_config: JEPAConfig,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.jepa_config = jepa_config

    def forward(
        self,
        x_t: torch.Tensor,
        x_t_plus_1: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute JEPA loss and predicted next latent map.

        Args:
            x_t: Current observation [B, C_in, H, W].
            x_t_plus_1: Next observation [B, C_in, H, W] (target for prediction).
            action: Action taken [B, A_dim].

        Returns:
            loss: Scalar loss (consistency + variance regularization).
            prediction: Predicted next latent map [B, C, H', W'].
        """
        # Source encoding (with grad) for predictor input
        z_t = self.encoder(x_t)
        # Target next state (no grad)
        with torch.no_grad():
            z_next_target = self.encoder(x_t_plus_1)

        prediction = self.predictor(z_t, action)

        loss_consistency = latent_consistency_loss(prediction, z_next_target)
        loss_var = variance_regularization(
            prediction,
            threshold=self.jepa_config.variance_threshold,
        )
        loss = loss_consistency + self.jepa_config.variance_weight * loss_var

        return loss, prediction

    def predict_next(
        self,
        x_t: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Return predicted next latent map given current observation and action."""
        z_t = self.encoder(x_t)
        return self.predictor(z_t, action)


def build_jepa(
    encoder_config: EncoderConfig,
    geometry_config: Optional[GeometryConfig],
    dynamics_config: DynamicsConfig,
    jepa_config: JEPAConfig,
) -> JEPA:
    """
    Build JEPA from configs. Encoder output_channels must match dynamics_config.latent_dim.
    """
    if encoder_config.output_channels != dynamics_config.latent_dim:
        raise ValueError(
            f"encoder_config.output_channels ({encoder_config.output_channels}) must equal "
            f"dynamics_config.latent_dim ({dynamics_config.latent_dim})."
        )
    encoder = SpatialEncoder(encoder_config, geometry_config)
    predictor = DynamicsPredictor(dynamics_config)
    return JEPA(encoder, predictor, jepa_config)
