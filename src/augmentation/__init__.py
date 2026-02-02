"""
NSCA Unified Augmentation Module.

Provides enhanced data augmentation for all modalities:
- Vision: RandAugment, MixUp/CutMix, temporal augmentation, physics-aware transforms
- Audio: SpecAugment, pitch shifting, room reverb, background mixing
- Proprioception: Sensor noise, joint dropout, temporal jitter

All augmentations support physics-aware mode that disables gravity-inconsistent
transformations (e.g., vertical flips) for physics prediction tasks.
"""

from .video_aug import (
    EnhancedVideoAugmentation,
    RandAugment,
    MixUp,
    CutMix,
    TemporalAugmentation,
)
from .audio_aug import (
    EnhancedAudioAugmentation,
    SpecAugment,
    PitchShift,
    RoomReverb,
    BackgroundMix,
)
from .proprio_aug import (
    ProprioceptionAugmentation,
    SensorNoiseAugmentation,
    JointDropout,
    TemporalJitter,
)
from .unified import (
    UnifiedAugmentation,
    AugmentationConfig,
    create_augmentation_pipeline,
)

__all__ = [
    # Video
    'EnhancedVideoAugmentation',
    'RandAugment',
    'MixUp',
    'CutMix',
    'TemporalAugmentation',
    # Audio
    'EnhancedAudioAugmentation',
    'SpecAugment',
    'PitchShift',
    'RoomReverb',
    'BackgroundMix',
    # Proprioception
    'ProprioceptionAugmentation',
    'SensorNoiseAugmentation',
    'JointDropout',
    'TemporalJitter',
    # Unified
    'UnifiedAugmentation',
    'AugmentationConfig',
    'create_augmentation_pipeline',
]
