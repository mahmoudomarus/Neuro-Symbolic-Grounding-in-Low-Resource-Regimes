"""
World model: sensory encoder, geometry, JEPA core, dynamics predictor,
temporal model, and unified world model.
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

# Lazy imports for modules with complex dependencies
# Import these directly when needed to avoid circular imports:
# from src.world_model.temporal_world_model import TemporalWorldModel
# from src.world_model.enhanced_dynamics import EnhancedDynamicsPredictor
# from src.world_model.unified_world_model import UnifiedWorldModel

__all__ = [
    # Original
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
