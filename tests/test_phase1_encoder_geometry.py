"""
Phase 1 tests: Sensory Encoder and Geometric Priors.
Verifies SpatialEncoder outputs [B, C, H, W] and RotaryEmbedding2D applies correctly.
"""
from __future__ import annotations

import torch

from src.world_model.config import EncoderConfig, GeometryConfig
from src.world_model.encoder import SpatialEncoder
from src.world_model.geometry import RotaryEmbedding2D


def test_rotary_embedding_2d_shape() -> None:
    """RotaryEmbedding2D preserves spatial map shape [B, C, H, W]."""
    cfg = GeometryConfig(dim=64, max_height=16, max_width=16)
    rope = RotaryEmbedding2D(cfg)
    x = torch.randn(2, 64, 8, 8)
    out = rope(x)
    assert out.shape == x.shape


def test_rotary_embedding_2d_different_positions() -> None:
    """Same content at different positions gets different rotations (relative position)."""
    cfg = GeometryConfig(dim=64, max_height=16, max_width=16)
    rope = RotaryEmbedding2D(cfg)
    patch = torch.randn(1, 64, 1, 1)
    # Place same patch at (0,0) in one map and at (1,1) in another; rest zeros
    x1 = torch.zeros(1, 64, 8, 8)
    x1[:, :, 0, 0] = patch.squeeze()
    x2 = torch.zeros(1, 64, 8, 8)
    x2[:, :, 1, 1] = patch.squeeze()
    out1 = rope(x1)
    out2 = rope(x2)
    # Output at (0,0) vs (1,1) should differ due to position-dependent rotation
    assert not torch.allclose(out1[:, :, 0, 0], out2[:, :, 1, 1])


def test_spatial_encoder_output_shape() -> None:
    """SpatialEncoder outputs SpatialMap [B, C, H, W], not a flat vector."""
    enc_cfg = EncoderConfig(
        input_channels=3,
        base_channels=32,
        num_blocks=2,
        output_channels=64,
        strides_per_stage=(2, 2, 2),
        use_geometry=True,
    )
    geom_cfg = GeometryConfig(dim=64, max_height=32, max_width=32)
    encoder = SpatialEncoder(enc_cfg, geom_cfg)
    x = torch.randn(2, 3, 64, 64)
    z = encoder(x)
    assert z.dim() == 4
    assert z.shape[0] == 2
    assert z.shape[1] == 64
    assert z.shape[2] < 64 and z.shape[3] < 64


def test_spatial_encoder_without_geometry() -> None:
    """SpatialEncoder runs with use_geometry=False (no RotaryEmbedding2D)."""
    enc_cfg = EncoderConfig(
        input_channels=3,
        base_channels=32,
        num_blocks=2,
        output_channels=64,
        strides_per_stage=(2, 2),
        use_geometry=False,
    )
    encoder = SpatialEncoder(enc_cfg, geometry_config=None)
    x = torch.randn(1, 3, 32, 32)
    z = encoder(x)
    assert z.shape[1] == 64 and z.dim() == 4
