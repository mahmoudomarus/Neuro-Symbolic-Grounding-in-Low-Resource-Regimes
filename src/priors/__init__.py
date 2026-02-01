"""
Innate sensory priors: The "born with" knowledge that structures perception.

These are fixed, non-learned transformations that mirror biological sensory processing:
- Visual: Gabor filterbank (like retinal/V1 orientation columns)
- Audio: Mel-frequency spectrogram (like cochlear frequency separation)
- Spatial: 3D rotary position encoding (like vestibular spatial awareness)
"""

from .visual_prior import GaborFilterbank, apply_gabor_filters
from .audio_prior import MelFilterbank, create_mel_spectrogram
from .spatial_prior import RotaryEmbedding3D, apply_3d_rotary

__all__ = [
    "GaborFilterbank",
    "apply_gabor_filters",
    "MelFilterbank",
    "create_mel_spectrogram",
    "RotaryEmbedding3D",
    "apply_3d_rotary",
]
