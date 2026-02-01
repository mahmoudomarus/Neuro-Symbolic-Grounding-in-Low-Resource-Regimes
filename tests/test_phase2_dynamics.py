"""
Phase 2 tests: Dynamics predictor and JEPA core.

Verifies predictor(z, action) returns same shape as z, loss is non-zero for
random inputs, and spatial [H, W] is maintained throughout.
"""
from __future__ import annotations

import torch

from src.world_model.config import (
    DynamicsConfig,
    EncoderConfig,
    GeometryConfig,
    JEPAConfig,
)
from src.world_model.dynamics import DynamicsPredictor
from src.world_model.jepa_core import JEPA, build_jepa
from src.world_model.encoder import SpatialEncoder


def test_predictor_output_shape() -> None:
    """DynamicsPredictor(z, action) returns a tensor of the same shape as z."""
    cfg = DynamicsConfig(
        action_dim=4,
        latent_dim=64,
        hidden_dim=128,
        num_layers=2,
    )
    predictor = DynamicsPredictor(cfg)
    B, C, H, W = 2, 64, 8, 8
    z = torch.randn(B, C, H, W)
    action = torch.randn(B, cfg.action_dim)
    out = predictor(z, action)
    assert out.shape == z.shape
    assert out.shape == (B, C, H, W)


def test_predictor_spatial_preserved() -> None:
    """Spatial [H, W] dimensions are preserved; no flattening."""
    cfg = DynamicsConfig(
        action_dim=8,
        latent_dim=32,
        hidden_dim=64,
        num_layers=3,
    )
    predictor = DynamicsPredictor(cfg)
    H, W = 12, 16
    z = torch.randn(1, 32, H, W)
    action = torch.randn(1, 8)
    out = predictor(z, action)
    assert out.dim() == 4
    assert out.shape[2] == H and out.shape[3] == W


def test_jepa_loss_not_zero_random_inputs() -> None:
    """JEPA loss is non-zero for random inputs (consistency + variance terms)."""
    enc_cfg = EncoderConfig(
        input_channels=3,
        base_channels=32,
        num_blocks=2,
        output_channels=64,
        strides_per_stage=(2, 2, 2),
        use_geometry=False,
    )
    dyn_cfg = DynamicsConfig(
        action_dim=4,
        latent_dim=64,
        hidden_dim=128,
        num_layers=2,
    )
    jepa_cfg = JEPAConfig(learning_rate=1e-4, variance_weight=0.1)
    encoder = SpatialEncoder(enc_cfg, geometry_config=None)
    predictor = DynamicsPredictor(dyn_cfg)
    jepa = JEPA(encoder, predictor, jepa_cfg)

    B = 4
    x_t = torch.randn(B, 3, 32, 32)
    x_t_plus_1 = torch.randn(B, 3, 32, 32)
    action = torch.randn(B, 4)

    loss, prediction = jepa(x_t, x_t_plus_1, action)
    assert loss.dim() == 0
    assert loss.item() > 0
    assert prediction.shape[0] == B
    assert prediction.shape[1] == 64
    assert prediction.dim() == 4


def test_build_jepa_and_forward() -> None:
    """build_jepa produces a model and forward returns loss + prediction."""
    enc_cfg = EncoderConfig(
        input_channels=3,
        base_channels=16,
        num_blocks=1,
        output_channels=32,
        strides_per_stage=(2, 2),
        use_geometry=False,
    )
    geom_cfg = None
    dyn_cfg = DynamicsConfig(
        action_dim=2,
        latent_dim=32,
        hidden_dim=64,
        num_layers=2,
    )
    jepa_cfg = JEPAConfig(learning_rate=1e-4, variance_weight=0.5)
    jepa = build_jepa(enc_cfg, geom_cfg, dyn_cfg, jepa_cfg)

    x_t = torch.randn(2, 3, 16, 16)
    x_t_plus_1 = torch.randn(2, 3, 16, 16)
    action = torch.randn(2, 2)
    loss, pred = jepa(x_t, x_t_plus_1, action)
    assert loss.requires_grad
    assert pred.shape[1] == 32
    assert pred.shape[2] == 4 and pred.shape[3] == 4  # 16/2/2
