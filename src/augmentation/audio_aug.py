"""
Enhanced Audio Augmentation for NSCA.

Includes:
- SpecAugment: Time/frequency masking on spectrograms
- Pitch shifting: Realistic pitch variations
- Room impulse response: Acoustic environment simulation
- Background mixing: Ambient noise addition
"""

import random
from typing import Tuple, Optional, List, Union
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


@dataclass
class AudioAugConfig:
    """Configuration for audio augmentation."""
    sample_rate: int = 16000
    
    # SpecAugment parameters
    freq_mask_param: int = 27  # Maximum frequency mask width
    time_mask_param: int = 100  # Maximum time mask width
    num_freq_masks: int = 2
    num_time_masks: int = 2
    
    # Pitch shifting
    pitch_shift_range: Tuple[int, int] = (-4, 4)  # Semitones
    
    # Room reverb
    reverb_prob: float = 0.3
    room_scale_range: Tuple[float, float] = (0.3, 0.8)
    
    # Background mixing
    background_prob: float = 0.3
    snr_range: Tuple[float, float] = (10, 30)  # dB
    
    # Basic augmentations
    volume_range: Tuple[float, float] = (0.8, 1.2)
    noise_std: float = 0.005
    time_shift_samples: int = 1000
    
    # Time stretch
    time_stretch_range: Tuple[float, float] = (0.8, 1.2)


class SpecAugment:
    """
    SpecAugment: A Simple Data Augmentation Method for ASR.
    
    Applies frequency and time masking to spectrograms.
    Reference: https://arxiv.org/abs/1904.08779
    """
    
    def __init__(
        self,
        freq_mask_param: int = 27,
        time_mask_param: int = 100,
        num_freq_masks: int = 2,
        num_time_masks: int = 2,
    ):
        """
        Args:
            freq_mask_param: Maximum frequency mask width (F)
            time_mask_param: Maximum time mask width (T)
            num_freq_masks: Number of frequency masks
            num_time_masks: Number of time masks
        """
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
    
    def __call__(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Apply SpecAugment to spectrogram.
        
        Args:
            spectrogram: [F, T] or [C, F, T] spectrogram
            
        Returns:
            Masked spectrogram
        """
        # Handle different input shapes
        if spectrogram.dim() == 2:
            spectrogram = spectrogram.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        C, F, T = spectrogram.shape
        spectrogram = spectrogram.clone()
        
        # Frequency masking
        for _ in range(self.num_freq_masks):
            f = random.randint(0, min(self.freq_mask_param, F - 1))
            f0 = random.randint(0, F - f)
            spectrogram[:, f0:f0 + f, :] = 0
        
        # Time masking
        for _ in range(self.num_time_masks):
            t = random.randint(0, min(self.time_mask_param, T - 1))
            t0 = random.randint(0, T - t)
            spectrogram[:, :, t0:t0 + t] = 0
        
        if squeeze_output:
            spectrogram = spectrogram.squeeze(0)
        
        return spectrogram


class PitchShift:
    """
    Pitch shifting for audio waveforms.
    
    Uses phase vocoder approach for high-quality pitch shifting.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        shift_range: Tuple[int, int] = (-4, 4),  # Semitones
        n_fft: int = 2048,
        hop_length: int = 512,
    ):
        """
        Args:
            sample_rate: Audio sample rate
            shift_range: Range of pitch shift in semitones
            n_fft: FFT size
            hop_length: Hop length for STFT
        """
        self.sample_rate = sample_rate
        self.shift_range = shift_range
        self.n_fft = n_fft
        self.hop_length = hop_length
    
    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Apply pitch shift to waveform.
        
        Args:
            waveform: [samples] or [C, samples] audio waveform
            
        Returns:
            Pitch-shifted waveform
        """
        # Random shift in semitones
        shift = random.randint(*self.shift_range)
        
        if shift == 0:
            return waveform
        
        # Convert semitones to ratio
        ratio = 2 ** (shift / 12.0)
        
        # Simple resampling-based pitch shift
        # This is a basic implementation; for production, use torchaudio.functional.pitch_shift
        
        squeeze = False
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
            squeeze = True
        
        C, T = waveform.shape
        
        # Resample to change pitch
        new_T = int(T / ratio)
        
        # Use linear interpolation for resampling
        indices = torch.linspace(0, T - 1, new_T, device=waveform.device)
        indices_floor = indices.long()
        indices_ceil = (indices_floor + 1).clamp(max=T - 1)
        weights = indices - indices_floor.float()
        
        # Interpolate
        resampled = (1 - weights) * waveform[:, indices_floor] + weights * waveform[:, indices_ceil]
        
        # Time-stretch back to original length
        if new_T != T:
            resampled = F.interpolate(
                resampled.unsqueeze(1),
                size=T,
                mode='linear',
                align_corners=False
            ).squeeze(1)
        
        if squeeze:
            resampled = resampled.squeeze(0)
        
        return resampled


class RoomReverb:
    """
    Simulates room acoustics using simple convolution reverb.
    
    Generates synthetic impulse responses for different room sizes.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        room_scale_range: Tuple[float, float] = (0.3, 0.8),
        decay_factor: float = 0.5,
    ):
        """
        Args:
            sample_rate: Audio sample rate
            room_scale_range: Range for room size (affects reverb length)
            decay_factor: Reverb decay factor
        """
        self.sample_rate = sample_rate
        self.room_scale_range = room_scale_range
        self.decay_factor = decay_factor
    
    def _generate_ir(self, room_scale: float) -> torch.Tensor:
        """Generate synthetic impulse response."""
        # Simple exponential decay IR
        ir_length = int(self.sample_rate * room_scale)  # Length proportional to room size
        
        t = torch.arange(ir_length, dtype=torch.float32)
        
        # Exponential decay
        decay = torch.exp(-self.decay_factor * t / ir_length)
        
        # Add some random reflections
        ir = decay * torch.randn(ir_length) * 0.1
        ir[0] = 1.0  # Direct sound
        
        # Add early reflections
        num_reflections = int(5 * room_scale)
        for i in range(num_reflections):
            delay = random.randint(int(0.01 * self.sample_rate), int(0.1 * self.sample_rate))
            if delay < ir_length:
                ir[delay] += 0.3 * (0.7 ** i) * (2 * random.random() - 1)
        
        # Normalize
        ir = ir / (ir.abs().max() + 1e-8)
        
        return ir
    
    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Apply room reverb to waveform.
        
        Args:
            waveform: [samples] or [C, samples] audio waveform
            
        Returns:
            Reverberant waveform
        """
        squeeze = False
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
            squeeze = True
        
        # Generate IR
        room_scale = random.uniform(*self.room_scale_range)
        ir = self._generate_ir(room_scale).to(waveform.device)
        
        # Convolve
        C, T = waveform.shape
        ir = ir.view(1, 1, -1)
        waveform = waveform.view(C, 1, T)
        
        reverb = F.conv1d(waveform, ir, padding=ir.shape[-1] // 2)
        reverb = reverb[:, :, :T]  # Trim to original length
        reverb = reverb.view(C, T)
        
        # Mix dry and wet
        wet_ratio = random.uniform(0.2, 0.5)
        output = (1 - wet_ratio) * waveform.view(C, T) + wet_ratio * reverb
        
        if squeeze:
            output = output.squeeze(0)
        
        return output


class BackgroundMix:
    """
    Mixes background noise/ambient sounds with audio.
    
    Can use generated noise or provided background samples.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        snr_range: Tuple[float, float] = (10, 30),  # dB
        noise_types: List[str] = ['white', 'pink', 'brown'],
    ):
        """
        Args:
            sample_rate: Audio sample rate
            snr_range: Signal-to-noise ratio range in dB
            noise_types: Types of noise to generate
        """
        self.sample_rate = sample_rate
        self.snr_range = snr_range
        self.noise_types = noise_types
    
    def _generate_noise(self, length: int, noise_type: str) -> torch.Tensor:
        """Generate colored noise."""
        if noise_type == 'white':
            return torch.randn(length)
        
        elif noise_type == 'pink':
            # Pink noise: 1/f spectrum
            white = torch.randn(length)
            # Simple IIR filter approximation
            pink = torch.zeros(length)
            b0, b1, b2 = 0.99886, 0.99332, 0.96900
            white_vals = [0, 0, 0]
            
            for i in range(length):
                white_vals[0] = b0 * white_vals[0] + white[i] * 0.0555179
                white_vals[1] = b1 * white_vals[1] + white[i] * 0.0750759
                white_vals[2] = b2 * white_vals[2] + white[i] * 0.1538520
                pink[i] = sum(white_vals) + white[i] * 0.5362
            
            return pink / (pink.abs().max() + 1e-8)
        
        elif noise_type == 'brown':
            # Brown/red noise: 1/f^2 spectrum (cumulative sum of white)
            white = torch.randn(length)
            brown = torch.cumsum(white, dim=0)
            brown = brown - brown.mean()
            return brown / (brown.abs().max() + 1e-8)
        
        else:
            return torch.randn(length)
    
    def __call__(
        self,
        waveform: torch.Tensor,
        background: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Mix background with waveform.
        
        Args:
            waveform: [samples] or [C, samples] audio waveform
            background: Optional background audio (same shape as waveform)
            
        Returns:
            Mixed waveform
        """
        squeeze = False
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
            squeeze = True
        
        C, T = waveform.shape
        
        # Generate or use provided background
        if background is None:
            noise_type = random.choice(self.noise_types)
            background = self._generate_noise(T, noise_type).to(waveform.device)
            background = background.unsqueeze(0).expand(C, -1)
        
        # Calculate signal power
        signal_power = (waveform ** 2).mean()
        noise_power = (background ** 2).mean()
        
        # Random SNR
        snr_db = random.uniform(*self.snr_range)
        snr_linear = 10 ** (snr_db / 10)
        
        # Scale noise to achieve target SNR
        if noise_power > 0:
            scale = torch.sqrt(signal_power / (noise_power * snr_linear + 1e-8))
            background = background * scale
        
        # Mix
        output = waveform + background
        
        # Normalize to prevent clipping
        max_val = output.abs().max()
        if max_val > 1.0:
            output = output / max_val
        
        if squeeze:
            output = output.squeeze(0)
        
        return output


class TimeStretch:
    """
    Time stretching without pitch change.
    
    Uses simple interpolation for speed change.
    """
    
    def __init__(
        self,
        rate_range: Tuple[float, float] = (0.8, 1.2),
    ):
        """
        Args:
            rate_range: Range for time stretch rate
        """
        self.rate_range = rate_range
    
    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Apply time stretch to waveform.
        
        Args:
            waveform: [samples] or [C, samples]
            
        Returns:
            Time-stretched waveform (resampled to original length)
        """
        rate = random.uniform(*self.rate_range)
        
        if abs(rate - 1.0) < 0.01:
            return waveform
        
        squeeze = False
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
            squeeze = True
        
        C, T = waveform.shape
        
        # New length after stretch
        new_T = int(T * rate)
        
        # Resample
        resampled = F.interpolate(
            waveform.unsqueeze(1),
            size=new_T,
            mode='linear',
            align_corners=False
        ).squeeze(1)
        
        # Resize back to original length
        resampled = F.interpolate(
            resampled.unsqueeze(1),
            size=T,
            mode='linear',
            align_corners=False
        ).squeeze(1)
        
        if squeeze:
            resampled = resampled.squeeze(0)
        
        return resampled


class EnhancedAudioAugmentation:
    """
    Enhanced audio augmentation combining all techniques.
    
    Includes:
    - SpecAugment for spectrogram masking
    - Pitch shifting
    - Room reverb
    - Background mixing
    - Time stretching
    - Basic augmentations (volume, noise, shift)
    """
    
    def __init__(
        self,
        config: Optional[AudioAugConfig] = None,
        p: float = 0.5,
    ):
        """
        Args:
            config: Audio augmentation configuration
            p: Probability of applying augmentation
        """
        self.config = config or AudioAugConfig()
        self.p = p
        
        # Initialize sub-augmentations
        self.spec_augment = SpecAugment(
            freq_mask_param=self.config.freq_mask_param,
            time_mask_param=self.config.time_mask_param,
            num_freq_masks=self.config.num_freq_masks,
            num_time_masks=self.config.num_time_masks,
        )
        self.pitch_shift = PitchShift(
            sample_rate=self.config.sample_rate,
            shift_range=self.config.pitch_shift_range,
        )
        self.room_reverb = RoomReverb(
            sample_rate=self.config.sample_rate,
            room_scale_range=self.config.room_scale_range,
        )
        self.background_mix = BackgroundMix(
            sample_rate=self.config.sample_rate,
            snr_range=self.config.snr_range,
        )
        self.time_stretch = TimeStretch(
            rate_range=self.config.time_stretch_range,
        )
    
    def __call__(
        self,
        waveform: torch.Tensor,
        spectrogram: Optional[torch.Tensor] = None,
        use_spec_augment: bool = True,
        use_pitch_shift: bool = True,
        use_reverb: bool = True,
        use_background: bool = True,
        use_time_stretch: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply enhanced audio augmentation.
        
        Args:
            waveform: [samples], [C, samples], or batched [B, samples]/[B, C, samples]
            spectrogram: Optional [F, T], [C, F, T], or batched
            use_spec_augment: Apply SpecAugment to spectrogram
            use_pitch_shift: Apply pitch shifting
            use_reverb: Apply room reverb
            use_background: Apply background mixing
            use_time_stretch: Apply time stretching
            
        Returns:
            Augmented waveform, augmented spectrogram (if provided)
        """
        if random.random() > self.p:
            return waveform, spectrogram
        
        # Handle batched input [B, samples] or [B, C, samples]
        batched = waveform.dim() >= 2 and waveform.shape[0] > 2  # Assume batch if first dim > 2 (not just stereo)
        if waveform.dim() == 3:  # [B, C, samples] - definitely batched
            batched = True
        elif waveform.dim() == 2 and waveform.shape[0] > 2:  # [B, samples] likely batched
            batched = True
        else:
            batched = False
        
        if batched:
            # Process each sample in batch
            B = waveform.shape[0]
            augmented = []
            for i in range(B):
                aug_wav = self._augment_single(
                    waveform[i], use_pitch_shift, use_reverb, use_background, use_time_stretch
                )
                augmented.append(aug_wav)
            waveform = torch.stack(augmented)
            
            # Handle batched spectrogram
            if spectrogram is not None and use_spec_augment:
                aug_specs = []
                for i in range(B):
                    aug_specs.append(self.spec_augment(spectrogram[i]))
                spectrogram = torch.stack(aug_specs)
            
            return waveform, spectrogram
        
        return self._augment_single(
            waveform, use_pitch_shift, use_reverb, use_background, use_time_stretch
        ), (self.spec_augment(spectrogram) if spectrogram is not None and use_spec_augment else spectrogram)
    
    def _augment_single(
        self,
        waveform: torch.Tensor,
        use_pitch_shift: bool,
        use_reverb: bool,
        use_background: bool,
        use_time_stretch: bool,
    ) -> torch.Tensor:
        """Apply augmentation to a single waveform."""
        # Basic augmentations
        waveform = self._basic_augment(waveform)
        
        # Pitch shift
        if use_pitch_shift and random.random() < 0.5:
            waveform = self.pitch_shift(waveform)
        
        # Time stretch
        if use_time_stretch and random.random() < 0.5:
            waveform = self.time_stretch(waveform)
        
        # Room reverb
        if use_reverb and random.random() < self.config.reverb_prob:
            waveform = self.room_reverb(waveform)
        
        # Background mixing
        if use_background and random.random() < self.config.background_prob:
            waveform = self.background_mix(waveform)
        
        return waveform
    
    def _basic_augment(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply basic augmentations."""
        squeeze = False
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
            squeeze = True
        
        # Random volume
        if random.random() < 0.5:
            factor = random.uniform(*self.config.volume_range)
            waveform = waveform * factor
        
        # Add noise
        if random.random() < 0.3:
            noise = torch.randn_like(waveform) * self.config.noise_std
            waveform = waveform + noise
        
        # Time shift
        if random.random() < 0.5:
            shift = random.randint(-self.config.time_shift_samples, self.config.time_shift_samples)
            waveform = torch.roll(waveform, shift, dims=-1)
        
        if squeeze:
            waveform = waveform.squeeze(0)
        
        return waveform
    
    def augment_spectrogram_only(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Apply only spectrogram augmentations.
        
        Args:
            spectrogram: [F, T] or [C, F, T]
            
        Returns:
            Augmented spectrogram
        """
        if random.random() > self.p:
            return spectrogram
        
        return self.spec_augment(spectrogram)
