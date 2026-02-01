"""
Multi-modal encoders module.

Each encoder applies innate priors BEFORE learned features,
mimicking how biological sensory systems work.
"""

from .vision_encoder import VisionEncoderWithPriors, VisionEncoderConfig
from .audio_encoder import AudioEncoderWithPriors, AudioEncoderConfig
from .proprio_encoder import ProprioEncoder, ProprioEncoderConfig

__all__ = [
    "VisionEncoderWithPriors",
    "VisionEncoderConfig",
    "AudioEncoderWithPriors", 
    "AudioEncoderConfig",
    "ProprioEncoder",
    "ProprioEncoderConfig",
]
