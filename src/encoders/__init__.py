"""
Multi-modal encoders: Convert raw sensory input to rich feature representations.

Uses pre-trained backbone models (frozen):
- Vision: DINO ViT-B/16 (facebook/dino-vitb16)
- Audio: Whisper Encoder (openai/whisper-base)
- Proprioception: Custom MLP (trained from scratch, low dimensional)

All encoders project to a shared 512-dimensional space for cross-modal fusion.
"""

from .vision_encoder import DinoVisionEncoder
from .audio_encoder import WhisperAudioEncoder
from .proprio_encoder import ProprioEncoder
from .projection import ModalityProjector

__all__ = [
    "DinoVisionEncoder",
    "WhisperAudioEncoder", 
    "ProprioEncoder",
    "ModalityProjector",
]
