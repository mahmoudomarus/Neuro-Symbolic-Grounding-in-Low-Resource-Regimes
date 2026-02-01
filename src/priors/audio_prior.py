"""
Auditory spectral prior: Mel-frequency decomposition mimicking cochlear processing.

Biological inspiration:
- Cochlea physically separates frequencies (basilar membrane)
- Different positions respond to different frequencies
- This is innate structure, not learned
- Mel scale approximates human frequency perception

Implementation:
- Fixed Mel filterbank transformation
- Converts waveform or STFT to Mel spectrogram
- Applied before any learnable audio encoder
- Output represents "what the auditory system hears"
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MelFilterbank(nn.Module):
    """
    Fixed Mel-filterbank for audio decomposition.
    
    Creates triangular filters spaced on the Mel scale:
    - Low frequencies: Linearly spaced (fine resolution)
    - High frequencies: Logarithmically spaced (coarse resolution)
    - Matches human auditory perception
    
    Analogous to cochlear frequency decomposition.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        n_mels: int = 80,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        hop_length: int = 320,  # 20ms at 16kHz
        win_length: int = 1024,  # 64ms at 16kHz
        power: float = 2.0,  # Magnitude spectrogram
    ):
        """
        Initialize Mel filterbank.
        
        Args:
            sample_rate: Audio sample rate in Hz (default 16000)
            n_fft: FFT size (default 1024)
            n_mels: Number of Mel bins (default 80)
            f_min: Minimum frequency in Hz
            f_max: Maximum frequency in Hz (default sample_rate//2)
            hop_length: Hop length for STFT (default 320 = 20ms @ 16kHz)
            win_length: Window length for STFT (default 1024)
            power: Power for spectrogram (2.0 = power, 1.0 = magnitude)
        """
        super().__init__()
        
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max or sample_rate // 2
        self.hop_length = hop_length
        self.win_length = win_length
        self.power = power
        
        # Create Hann window for STFT
        window = torch.hann_window(win_length)
        self.register_buffer("window", window)
        
        # Create Mel filterbank
        mel_fb = self._create_mel_filterbank()
        self.register_buffer("mel_fb", mel_fb)
        
    def _hz_to_mel(self, hz: float) -> float:
        """Convert Hz to Mel scale."""
        return 2595 * math.log10(1 + hz / 700)
    
    def _mel_to_hz(self, mel: float) -> float:
        """Convert Mel scale to Hz."""
        return 700 * (10 ** (mel / 2595) - 1)
    
    def _create_mel_filterbank(self) -> torch.Tensor:
        """Create Mel filterbank matrix: [n_mels, n_freqs]."""
        n_freqs = self.n_fft // 2 + 1
        
        # Mel points
        mel_min = self._hz_to_mel(self.f_min)
        mel_max = self._hz_to_mel(self.f_max)
        mels = torch.linspace(mel_min, mel_max, self.n_mels + 2)
        
        # Convert to frequency bins
        freqs = self._mel_to_hz(mels)
        fft_freqs = torch.linspace(0, self.sample_rate / 2, n_freqs)
        
        # Create filterbank
        mel_fb = torch.zeros(self.n_mels, n_freqs)
        
        for i in range(self.n_mels):
            # Triangular filters
            f_left = freqs[i]
            f_center = freqs[i + 1]
            f_right = freqs[i + 2]
            
            # Rising slope
            if f_center > f_left:
                left_slope = (fft_freqs - f_left) / (f_center - f_left)
                left_slope = torch.clamp(left_slope, 0, 1)
            else:
                left_slope = torch.zeros_like(fft_freqs)
            
            # Falling slope
            if f_right > f_center:
                right_slope = (f_right - fft_freqs) / (f_right - f_center)
                right_slope = torch.clamp(right_slope, 0, 1)
            else:
                right_slope = torch.zeros_like(fft_freqs)
            
            # Combine
            mel_fb[i] = torch.minimum(left_slope, right_slope)
        
        return mel_fb
    
    def _stft(self, waveform: torch.Tensor) -> torch.Tensor:
        """Compute Short-Time Fourier Transform."""
        # Pad to center windows
        pad_amount = self.n_fft // 2
        waveform = F.pad(waveform, (pad_amount, pad_amount), mode='reflect')
        
        # STFT
        spec = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
        )
        
        # Power spectrogram
        if self.power == 2.0:
            spec = spec.abs() ** 2
        elif self.power == 1.0:
            spec = spec.abs()
        else:
            spec = spec.abs() ** self.power
        
        return spec
    
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Convert waveform to Mel spectrogram.
        
        Args:
            waveform: [B, T] or [B, 1, T] where T is number of samples
            
        Returns:
            Mel spectrogram: [B, n_mels, T_frames]
        """
        # Handle input shape
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)  # [B, T]
        
        batch_size = waveform.shape[0]
        
        # Compute STFT
        spec = self._stft(waveform)  # [B, n_freqs, T_frames]
        
        # Apply Mel filterbank
        # spec: [B, n_freqs, T_frames]
        # mel_fb: [n_mels, n_freqs]
        mel_spec = torch.matmul(self.mel_fb, spec)  # [B, n_mels, T_frames]
        
        return mel_spec
    
    def forward_db(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Convert waveform to Mel spectrogram in dB scale.
        
        Args:
            waveform: [B, T] or [B, 1, T]
            
        Returns:
            Mel spectrogram in dB: [B, n_mels, T_frames]
        """
        mel_spec = self.forward(waveform)
        
        # Convert to dB: 10 * log10(mel_spec + epsilon)
        mel_spec_db = 10 * torch.log10(mel_spec + 1e-10)
        
        # Normalize to reasonable range (approx -80 to 0 dB)
        mel_spec_db = torch.clamp(mel_spec_db, min=-80.0)
        
        return mel_spec_db


def create_mel_spectrogram(
    waveform: torch.Tensor,
    sample_rate: int = 16000,
    n_mels: int = 80,
    hop_length: int = 320,
) -> torch.Tensor:
    """
    Convenience function to create Mel spectrogram.
    
    Args:
        waveform: [B, T] audio waveform
        sample_rate: Sample rate in Hz (default 16000)
        n_mels: Number of Mel bins (default 80)
        hop_length: Hop length for STFT (default 320 = 20ms @ 16kHz)
        
    Returns:
        Mel spectrogram in dB: [B, n_mels, T_frames]
    """
    mel_fb = MelFilterbank(
        sample_rate=sample_rate,
        n_fft=1024,
        n_mels=n_mels,
        hop_length=hop_length,
    )
    
    mel_fb = mel_fb.to(waveform.device)
    
    return mel_fb.forward_db(waveform)
