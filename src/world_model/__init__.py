"""
World model: sensory encoder, geometry, JEPA core, and dynamics predictor.
"""
from .config import (
    DynamicsConfig,
    EncoderConfig,
    GeometryConfig,
    JEPAConfig,
    TrainingConfig,
)
from .dynamics import DynamicsPredictor
from .encoder import SpatialEncoder
from .geometry import GeometricBias, RotaryEmbedding2D
from .jepa_core import JEPA, build_jepa

__all__ = [
    "DynamicsConfig",
    "EncoderConfig",
    "GeometryConfig",
    "JEPAConfig",
    "TrainingConfig",
    "SpatialEncoder",
    "GeometricBias",
    "RotaryEmbedding2D",
    "DynamicsPredictor",
    "JEPA",
    "build_jepa",
]
