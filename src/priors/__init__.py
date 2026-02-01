"""
Innate priors module - mathematical structures that exist before learning.

These priors encode fundamental properties of perception that humans have from birth:
- Visual: Color opponency, edge detection (Gabor), depth cues
- Audio: Cochlear-like mel processing, onset detection
- Spatial: 3D understanding from 2D, perspective, occlusion
- Temporal: Causality, recency, periodicity
"""

from .visual_prior import (
    ColorOpponencyPrior, 
    GaborPrior, 
    DepthCuesPrior,
    TextureGradientPrior,
)
from .audio_prior import (
    AuditoryPrior, 
    OnsetDetector,
    SpectralContrastPrior,
    PitchPrior,
)
from .spatial_prior import (
    SpatialPrior3D,
    OcclusionPrior,
    CenterSurroundPrior,
    GridCellPrior,
)
from .temporal_prior import (
    TemporalPrior,
    RelativeTemporalEncoding,
    TemporalConvolutionPrior,
    CausalityPrior,
    RhythmPrior,
)

__all__ = [
    # Visual
    "ColorOpponencyPrior",
    "GaborPrior", 
    "DepthCuesPrior",
    "TextureGradientPrior",
    # Audio
    "AuditoryPrior",
    "OnsetDetector",
    "SpectralContrastPrior",
    "PitchPrior",
    # Spatial
    "SpatialPrior3D",
    "OcclusionPrior",
    "CenterSurroundPrior",
    "GridCellPrior",
    # Temporal
    "TemporalPrior",
    "RelativeTemporalEncoding",
    "TemporalConvolutionPrior",
    "CausalityPrior",
    "RhythmPrior",
]
