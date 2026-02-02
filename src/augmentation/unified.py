"""
Unified Multi-Modal Augmentation for NSCA.

Provides coordinated augmentation across all modalities:
- Vision, Audio, Proprioception
- Synchronized random seeds for temporal alignment
- Physics-aware mode
- YAML configuration support
"""

import random
from typing import Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass, field

import torch
import yaml

from .video_aug import EnhancedVideoAugmentation, VideoAugConfig
from .audio_aug import EnhancedAudioAugmentation, AudioAugConfig
from .proprio_aug import ProprioceptionAugmentation, ProprioAugConfig


@dataclass
class AugmentationConfig:
    """
    Configuration for unified multi-modal augmentation.
    
    Supports loading from YAML configuration files.
    """
    # Global settings
    enabled: bool = True
    probability: float = 0.5
    physics_aware: bool = True  # Disable gravity-inconsistent transforms
    synchronized: bool = True  # Use same random seed across modalities
    
    # Per-modality configs
    vision: VideoAugConfig = field(default_factory=VideoAugConfig)
    audio: AudioAugConfig = field(default_factory=AudioAugConfig)
    proprio: ProprioAugConfig = field(default_factory=ProprioAugConfig)
    
    # Modality-specific enable flags
    enable_vision: bool = True
    enable_audio: bool = True
    enable_proprio: bool = True
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'AugmentationConfig':
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls.from_dict(config_dict.get('augmentation', config_dict))
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AugmentationConfig':
        """Create configuration from dictionary."""
        # Extract global settings
        enabled = config_dict.get('enabled', True)
        probability = config_dict.get('probability', 0.5)
        physics_aware = config_dict.get('physics_aware', True)
        synchronized = config_dict.get('synchronized', True)
        
        # Extract vision config
        vision_dict = config_dict.get('vision', {})
        vision_config = VideoAugConfig(
            rand_augment_n=vision_dict.get('rand_augment', {}).get('n', 2),
            rand_augment_m=vision_dict.get('rand_augment', {}).get('m', 9),
            mixup_alpha=vision_dict.get('mixup_alpha', 0.2),
            cutmix_alpha=vision_dict.get('cutmix_alpha', 1.0),
            mix_prob=vision_dict.get('mix_prob', 0.5),
            temporal_dropout=vision_dict.get('temporal_dropout', 0.1),
            physics_aware=physics_aware,
            horizontal_flip_p=vision_dict.get('horizontal_flip_p', 0.5),
            brightness_range=tuple(vision_dict.get('brightness_range', [0.8, 1.2])),
            contrast_range=tuple(vision_dict.get('contrast_range', [0.8, 1.2])),
            crop_scale=tuple(vision_dict.get('crop_scale', [0.85, 1.0])),
            color_jitter_p=vision_dict.get('color_jitter_p', 0.8),
            grayscale_p=vision_dict.get('grayscale_p', 0.2),
        )
        
        # Extract audio config
        audio_dict = config_dict.get('audio', {})
        spec_aug = audio_dict.get('spec_augment', {})
        audio_config = AudioAugConfig(
            sample_rate=audio_dict.get('sample_rate', 16000),
            freq_mask_param=spec_aug.get('freq_mask', 27),
            time_mask_param=spec_aug.get('time_mask', 100),
            pitch_shift_range=tuple(audio_dict.get('pitch_shift', [-4, 4])),
            reverb_prob=audio_dict.get('reverb_prob', 0.3),
            background_prob=audio_dict.get('background_prob', 0.3),
            snr_range=tuple(audio_dict.get('snr_range', [10, 30])),
            time_stretch_range=tuple(audio_dict.get('time_stretch', [0.8, 1.2])),
        )
        
        # Extract proprio config
        proprio_dict = config_dict.get('proprio', {})
        proprio_config = ProprioAugConfig(
            noise_std=proprio_dict.get('noise_std', 0.01),
            joint_dropout_p=proprio_dict.get('joint_dropout', 0.1),
            temporal_jitter_ms=proprio_dict.get('temporal_jitter_ms', 10),
            position_noise=proprio_dict.get('position_noise', 0.005),
            velocity_noise=proprio_dict.get('velocity_noise', 0.01),
            drift_prob=proprio_dict.get('drift_prob', 0.1),
        )
        
        return cls(
            enabled=enabled,
            probability=probability,
            physics_aware=physics_aware,
            synchronized=synchronized,
            vision=vision_config,
            audio=audio_config,
            proprio=proprio_config,
            enable_vision=config_dict.get('enable_vision', True),
            enable_audio=config_dict.get('enable_audio', True),
            enable_proprio=config_dict.get('enable_proprio', True),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'enabled': self.enabled,
            'probability': self.probability,
            'physics_aware': self.physics_aware,
            'synchronized': self.synchronized,
            'enable_vision': self.enable_vision,
            'enable_audio': self.enable_audio,
            'enable_proprio': self.enable_proprio,
            'vision': {
                'rand_augment': {
                    'n': self.vision.rand_augment_n,
                    'm': self.vision.rand_augment_m,
                },
                'mixup_alpha': self.vision.mixup_alpha,
                'cutmix_alpha': self.vision.cutmix_alpha,
                'temporal_dropout': self.vision.temporal_dropout,
                'physics_aware': self.vision.physics_aware,
            },
            'audio': {
                'sample_rate': self.audio.sample_rate,
                'spec_augment': {
                    'freq_mask': self.audio.freq_mask_param,
                    'time_mask': self.audio.time_mask_param,
                },
                'pitch_shift': list(self.audio.pitch_shift_range),
            },
            'proprio': {
                'noise_std': self.proprio.noise_std,
                'joint_dropout': self.proprio.joint_dropout_p,
                'temporal_jitter_ms': self.proprio.temporal_jitter_ms,
            },
        }


class UnifiedAugmentation:
    """
    Unified multi-modal augmentation pipeline.
    
    Coordinates augmentation across vision, audio, and proprioception
    with optional synchronization for temporal alignment.
    """
    
    def __init__(
        self,
        config: Optional[AugmentationConfig] = None,
    ):
        """
        Args:
            config: Unified augmentation configuration
        """
        self.config = config or AugmentationConfig()
        
        # Initialize per-modality augmenters
        self.video_aug = EnhancedVideoAugmentation(
            config=self.config.vision,
            physics_aware=self.config.physics_aware,
            p=self.config.probability,
        )
        self.audio_aug = EnhancedAudioAugmentation(
            config=self.config.audio,
            p=self.config.probability,
        )
        self.proprio_aug = ProprioceptionAugmentation(
            config=self.config.proprio,
            p=self.config.probability,
        )
    
    def __call__(
        self,
        vision: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
        spectrogram: Optional[torch.Tensor] = None,
        vision2: Optional[torch.Tensor] = None,
        label: Optional[torch.Tensor] = None,
        label2: Optional[torch.Tensor] = None,
        use_mix: bool = False,
    ) -> Dict[str, Any]:
        """
        Apply unified augmentation to all modalities.
        
        Args:
            vision: Video frames [T, C, H, W] or [B, T, C, H, W]
            audio: Audio waveform [samples] or [C, samples]
            proprio: Proprioception [T, D] or [B, T, D]
            spectrogram: Optional spectrogram for SpecAugment
            vision2: Optional second video for MixUp/CutMix
            label: Optional label for primary sample
            label2: Optional label for second sample
            use_mix: Whether to apply MixUp/CutMix
            
        Returns:
            Dictionary with augmented modalities and metadata
        """
        if not self.config.enabled:
            return {
                'vision': vision,
                'audio': audio,
                'proprio': proprio,
                'spectrogram': spectrogram,
                'label': label,
                'mix_lambda': 1.0,
            }
        
        # Set synchronized random seed if enabled
        if self.config.synchronized:
            seed = random.randint(0, 2**32 - 1)
        
        result = {
            'mix_lambda': 1.0,
            'label': label,
        }
        
        # Augment vision
        if vision is not None and self.config.enable_vision:
            if self.config.synchronized:
                random.seed(seed)
                torch.manual_seed(seed)
            
            aug_vision, aug_label, mix_lambda = self.video_aug(
                vision,
                frames2=vision2,
                label=label,
                label2=label2,
                use_mix=use_mix,
            )
            result['vision'] = aug_vision
            result['label'] = aug_label
            result['mix_lambda'] = mix_lambda
        else:
            result['vision'] = vision
        
        # Augment audio
        if audio is not None and self.config.enable_audio:
            if self.config.synchronized:
                random.seed(seed)
                torch.manual_seed(seed)
            
            aug_audio, aug_spec = self.audio_aug(
                audio,
                spectrogram=spectrogram,
            )
            result['audio'] = aug_audio
            result['spectrogram'] = aug_spec
        else:
            result['audio'] = audio
            result['spectrogram'] = spectrogram
        
        # Augment proprioception
        if proprio is not None and self.config.enable_proprio:
            if self.config.synchronized:
                random.seed(seed)
                torch.manual_seed(seed)
            
            aug_proprio = self.proprio_aug(proprio)
            result['proprio'] = aug_proprio
        else:
            result['proprio'] = proprio
        
        return result
    
    def augment_vision(
        self,
        frames: torch.Tensor,
        frames2: Optional[torch.Tensor] = None,
        label: Optional[torch.Tensor] = None,
        label2: Optional[torch.Tensor] = None,
        use_mix: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], float]:
        """
        Augment only vision modality.
        
        Args:
            frames: Video frames [T, C, H, W]
            frames2: Optional second video for mixing
            label: Optional label
            label2: Optional second label
            use_mix: Apply MixUp/CutMix
            
        Returns:
            Augmented frames, optional label, mix lambda
        """
        if not self.config.enabled or not self.config.enable_vision:
            return frames, label, 1.0
        
        return self.video_aug(
            frames,
            frames2=frames2,
            label=label,
            label2=label2,
            use_mix=use_mix,
        )
    
    def augment_audio(
        self,
        waveform: torch.Tensor,
        spectrogram: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Augment only audio modality.
        
        Args:
            waveform: Audio waveform
            spectrogram: Optional spectrogram
            
        Returns:
            Augmented waveform, augmented spectrogram
        """
        if not self.config.enabled or not self.config.enable_audio:
            return waveform, spectrogram
        
        return self.audio_aug(waveform, spectrogram)
    
    def augment_proprio(
        self,
        proprio: torch.Tensor,
    ) -> torch.Tensor:
        """
        Augment only proprioception modality.
        
        Args:
            proprio: Proprioception data
            
        Returns:
            Augmented proprioception
        """
        if not self.config.enabled or not self.config.enable_proprio:
            return proprio
        
        return self.proprio_aug(proprio)


def create_augmentation_pipeline(
    config_path: Optional[str] = None,
    config_dict: Optional[Dict[str, Any]] = None,
    physics_aware: bool = True,
    probability: float = 0.5,
) -> UnifiedAugmentation:
    """
    Factory function to create augmentation pipeline.
    
    Args:
        config_path: Path to YAML configuration file
        config_dict: Dictionary configuration (overrides config_path)
        physics_aware: Enable physics-aware mode
        probability: Augmentation probability
        
    Returns:
        Configured UnifiedAugmentation instance
    """
    if config_path is not None:
        config = AugmentationConfig.from_yaml(config_path)
    elif config_dict is not None:
        config = AugmentationConfig.from_dict(config_dict)
    else:
        config = AugmentationConfig(
            physics_aware=physics_aware,
            probability=probability,
        )
    
    return UnifiedAugmentation(config=config)


# Convenience functions for quick augmentation

def quick_video_augment(
    frames: torch.Tensor,
    physics_aware: bool = True,
    p: float = 0.5,
) -> torch.Tensor:
    """Quick video augmentation with default settings."""
    aug = EnhancedVideoAugmentation(physics_aware=physics_aware, p=p)
    frames, _, _ = aug(frames)
    return frames


def quick_audio_augment(
    waveform: torch.Tensor,
    p: float = 0.5,
) -> torch.Tensor:
    """Quick audio augmentation with default settings."""
    aug = EnhancedAudioAugmentation(p=p)
    waveform, _ = aug(waveform)
    return waveform


def quick_proprio_augment(
    proprio: torch.Tensor,
    p: float = 0.5,
) -> torch.Tensor:
    """Quick proprioception augmentation with default settings."""
    aug = ProprioceptionAugmentation(p=p)
    return aug(proprio)
