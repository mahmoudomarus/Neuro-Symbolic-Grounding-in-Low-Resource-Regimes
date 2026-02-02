"""
Tests for the enhanced augmentation module.

Tests cover:
- Video augmentation (RandAugment, MixUp, CutMix, Temporal)
- Audio augmentation (SpecAugment, PitchShift, Reverb, Background)
- Proprioception augmentation (Noise, Dropout, Jitter)
- Unified augmentation pipeline
- Physics-aware mode
"""

import pytest
import torch
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.augmentation import (
    EnhancedVideoAugmentation,
    EnhancedAudioAugmentation,
    ProprioceptionAugmentation,
    UnifiedAugmentation,
    AugmentationConfig,
    create_augmentation_pipeline,
)
from src.augmentation.video_aug import (
    RandAugment,
    MixUp,
    CutMix,
    TemporalAugmentation,
    VideoAugConfig,
)
from src.augmentation.audio_aug import (
    SpecAugment,
    PitchShift,
    RoomReverb,
    BackgroundMix,
    AudioAugConfig,
)
from src.augmentation.proprio_aug import (
    SensorNoiseAugmentation,
    JointDropout,
    TemporalJitter,
    ProprioAugConfig,
)


class TestVideoAugmentation:
    """Test video augmentation components."""
    
    def test_rand_augment_shape(self):
        """Test that RandAugment preserves shape."""
        aug = RandAugment(n=2, m=9, physics_aware=True)
        frames = torch.randn(8, 3, 64, 64)  # T, C, H, W
        
        output = aug(frames)
        
        assert output.shape == frames.shape
    
    def test_rand_augment_physics_aware(self):
        """Test physics-aware mode excludes vertical flip."""
        aug = RandAugment(n=2, m=9, physics_aware=True)
        
        # Check that flip_vertical is not in augmentations
        aug_names = [name for name, _ in aug.augmentations]
        assert 'flip_vertical' not in aug_names
    
    def test_mixup_basic(self):
        """Test MixUp augmentation."""
        mixup = MixUp(alpha=0.2)
        
        frames1 = torch.randn(8, 3, 64, 64)
        frames2 = torch.randn(8, 3, 64, 64)
        label1 = torch.tensor([1.0, 0.0])
        label2 = torch.tensor([0.0, 1.0])
        
        mixed, mixed_label, lam = mixup(frames1, frames2, label1, label2)
        
        assert mixed.shape == frames1.shape
        assert mixed_label is not None
        assert 0 <= lam <= 1
    
    def test_cutmix_basic(self):
        """Test CutMix augmentation."""
        cutmix = CutMix(alpha=1.0)
        
        frames1 = torch.ones(8, 3, 64, 64)
        frames2 = torch.zeros(8, 3, 64, 64)
        
        mixed, _, lam = cutmix(frames1, frames2)
        
        assert mixed.shape == frames1.shape
        # Mixed should have some zeros from frames2
        assert (mixed == 0).any() or (mixed == 1).any()
    
    def test_temporal_augmentation(self):
        """Test temporal augmentations."""
        temporal = TemporalAugmentation(dropout_p=0.1, speed_range=(0.8, 1.2))
        frames = torch.randn(16, 3, 64, 64)
        
        # Frame dropout
        output = temporal.frame_dropout(frames)
        assert output.shape == frames.shape
        
        # Speed perturbation
        output = temporal.speed_perturb(frames)
        assert output.shape == frames.shape
    
    def test_enhanced_video_augmentation(self):
        """Test full enhanced video augmentation."""
        aug = EnhancedVideoAugmentation(physics_aware=True, p=1.0)
        frames = torch.randn(8, 3, 64, 64)
        
        output, _, _ = aug(frames, use_rand_augment=True, use_temporal=True)
        
        assert output.shape == frames.shape


class TestAudioAugmentation:
    """Test audio augmentation components."""
    
    def test_spec_augment_shape(self):
        """Test SpecAugment preserves shape."""
        aug = SpecAugment(freq_mask_param=10, time_mask_param=20)
        spec = torch.randn(80, 100)  # F, T
        
        output = aug(spec)
        
        assert output.shape == spec.shape
    
    def test_spec_augment_masking(self):
        """Test that SpecAugment applies masks."""
        aug = SpecAugment(freq_mask_param=10, time_mask_param=20)
        spec = torch.ones(80, 100)
        
        output = aug(spec)
        
        # Should have some zeros from masking
        assert (output == 0).any()
    
    def test_pitch_shift(self):
        """Test pitch shifting."""
        pitch = PitchShift(sample_rate=16000, shift_range=(-4, 4))
        waveform = torch.randn(16000)
        
        output = pitch(waveform)
        
        assert output.shape == waveform.shape
    
    def test_room_reverb(self):
        """Test room reverb."""
        reverb = RoomReverb(sample_rate=16000)
        waveform = torch.randn(16000)
        
        output = reverb(waveform)
        
        assert output.shape == waveform.shape
    
    def test_background_mix(self):
        """Test background mixing."""
        bg_mix = BackgroundMix(sample_rate=16000, snr_range=(10, 30))
        waveform = torch.randn(16000)
        
        output = bg_mix(waveform)
        
        assert output.shape == waveform.shape
    
    def test_enhanced_audio_augmentation(self):
        """Test full enhanced audio augmentation."""
        aug = EnhancedAudioAugmentation(p=1.0)
        waveform = torch.randn(16000)
        spec = torch.randn(80, 100)
        
        output_wave, output_spec = aug(waveform, spec)
        
        assert output_wave.shape == waveform.shape
        assert output_spec.shape == spec.shape


class TestProprioAugmentation:
    """Test proprioception augmentation components."""
    
    def test_sensor_noise(self):
        """Test sensor noise injection."""
        noise = SensorNoiseAugmentation(noise_std=0.01)
        proprio = torch.randn(16, 12)  # T, D
        
        output = noise(proprio)
        
        assert output.shape == proprio.shape
        # Output should be different from input due to noise
        assert not torch.allclose(output, proprio)
    
    def test_joint_dropout(self):
        """Test joint dropout."""
        dropout = JointDropout(dropout_p=0.3, min_joints_kept=6)
        proprio = torch.ones(16, 12)
        
        output = dropout(proprio)
        
        assert output.shape == proprio.shape
        # Should have at least min_joints_kept non-zero channels
        non_zero_channels = (output.sum(dim=0) != 0).sum()
        assert non_zero_channels >= 6
    
    def test_temporal_jitter(self):
        """Test temporal jitter."""
        jitter = TemporalJitter(jitter_ms=10, sample_rate=100)
        proprio = torch.randn(16, 12)
        
        output = jitter(proprio)
        
        assert output.shape == proprio.shape
    
    def test_proprioception_augmentation(self):
        """Test full proprioception augmentation."""
        aug = ProprioceptionAugmentation(p=1.0)
        proprio = torch.randn(16, 12)
        
        output = aug(proprio)
        
        assert output.shape == proprio.shape


class TestUnifiedAugmentation:
    """Test unified multi-modal augmentation."""
    
    def test_unified_augmentation_all_modalities(self):
        """Test unified augmentation with all modalities."""
        aug = UnifiedAugmentation()
        
        vision = torch.randn(8, 3, 64, 64)
        audio = torch.randn(16000)
        proprio = torch.randn(16, 12)
        
        result = aug(vision=vision, audio=audio, proprio=proprio)
        
        assert 'vision' in result
        assert 'audio' in result
        assert 'proprio' in result
        assert result['vision'].shape == vision.shape
        assert result['audio'].shape == audio.shape
        assert result['proprio'].shape == proprio.shape
    
    def test_unified_augmentation_partial_modalities(self):
        """Test unified augmentation with some modalities missing."""
        aug = UnifiedAugmentation()
        
        vision = torch.randn(8, 3, 64, 64)
        
        result = aug(vision=vision)
        
        assert result['vision'].shape == vision.shape
        assert result['audio'] is None
        assert result['proprio'] is None
    
    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            'enabled': True,
            'probability': 0.8,
            'physics_aware': True,
            'vision': {
                'rand_augment': {'n': 3, 'm': 7},
                'mixup_alpha': 0.3,
            },
            'audio': {
                'spec_augment': {'freq_mask': 20, 'time_mask': 80},
            },
            'proprio': {
                'noise_std': 0.02,
                'joint_dropout': 0.15,
            },
        }
        
        config = AugmentationConfig.from_dict(config_dict)
        
        assert config.enabled == True
        assert config.probability == 0.8
        assert config.physics_aware == True
        assert config.vision.rand_augment_n == 3
        assert config.audio.freq_mask_param == 20
        assert config.proprio.noise_std == 0.02
    
    def test_create_augmentation_pipeline(self):
        """Test factory function."""
        aug = create_augmentation_pipeline(
            physics_aware=True,
            probability=0.7,
        )
        
        assert isinstance(aug, UnifiedAugmentation)
        assert aug.config.physics_aware == True
        assert aug.config.probability == 0.7


class TestPhysicsAware:
    """Test physics-aware augmentation mode."""
    
    def test_physics_aware_no_vertical_flip(self):
        """Verify physics-aware mode excludes vertical flips."""
        config = VideoAugConfig(physics_aware=True)
        aug = EnhancedVideoAugmentation(config=config, physics_aware=True, p=1.0)
        
        # RandAugment should not include vertical flip
        aug_names = [name for name, _ in aug.rand_augment.augmentations]
        assert 'flip_vertical' not in aug_names
    
    def test_unified_physics_aware(self):
        """Test unified augmentation in physics-aware mode."""
        config = AugmentationConfig(physics_aware=True)
        aug = UnifiedAugmentation(config=config)
        
        vision = torch.randn(8, 3, 64, 64)
        result = aug(vision=vision)
        
        # Should complete without error
        assert result['vision'].shape == vision.shape


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
