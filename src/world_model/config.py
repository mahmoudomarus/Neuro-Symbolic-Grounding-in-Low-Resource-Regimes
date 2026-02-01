"""
Configuration dataclasses for the world model. No magic numbers; all dimensions
and thresholds are centralized here.
"""
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class GeometryConfig:
    """Configuration for 2D geometric (rotary) position encoding."""

    dim: int  # Channel dimension (must be even for pairwise rotation)
    max_height: int
    max_width: int
    base_freq: float = 10_000.0  # RoPE base frequency (from RoFormer)
    axis_balance: float = 1.0  # Balance between height/width frequencies (1.0 = equal)


@dataclass(frozen=True)
class EncoderConfig:
    """Configuration for the spatial sensory encoder."""

    input_channels: int
    base_channels: int
    num_blocks: int  # Number of residual blocks
    output_channels: int
    # Spatial output size can be inferred from input size and total stride
    # e.g. input (B, C, 64, 64) -> after stride 2^3 -> (B, C', 8, 8)
    strides_per_stage: Tuple[int, ...]  # Stride per stage, e.g. (2, 2, 2)
    use_geometry: bool = True  # Whether to apply RotaryEmbedding2D


@dataclass(frozen=True)
class DynamicsConfig:
    """Configuration for the dynamics predictor (imagination engine)."""

    action_dim: int
    latent_dim: int  # Must match encoder output_channels (C in [B, C, H, W])
    hidden_dim: int  # Channels in ResBlock stack
    num_layers: int  # Number of ResBlocks


@dataclass(frozen=True)
class JEPAConfig:
    """Configuration for JEPA training and loss."""

    learning_rate: float
    variance_weight: float  # Weight for variance regularization (anti-collapse, VICReg-style)
    variance_threshold: float = 1.0  # Target std above which no penalty


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for SimCLR kindergarten training (contrastive learning)."""

    batch_size: int = 128
    epochs: int = 5
    learning_rate: float = 3e-4
    temperature: float = 0.5  # For NT-Xent contrastive loss
    projection_dim: int = 128  # SimCLR projection head output
    dataset_path: str = "./data"


@dataclass(frozen=True)
class RealWorldConfig:
    """Configuration for real-world RGB image processing.
    
    Used when loading custom datasets (e.g., ./data/my_dataset) instead of
    the demo Fashion-MNIST. When present, EncoderConfig is dynamically built
    from these values to handle 3-channel RGB images.
    """

    input_resolution: int = 64  # Target spatial size after resize/crop
    input_channels: int = 3  # RGB channels
    patch_size: int = 8  # Patch size for spatial reasoning (not used directly in encoder)
    
    # ImageNet normalization statistics for RGB data
    normalize_mean: Tuple[float, ...] = (0.485, 0.456, 0.406)
    normalize_std: Tuple[float, ...] = (0.229, 0.224, 0.225)
