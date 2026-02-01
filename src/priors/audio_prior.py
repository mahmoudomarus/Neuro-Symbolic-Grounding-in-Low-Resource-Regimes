"""
Auditory innate priors - structures that exist in human hearing from birth.

These are NOT learned features but mathematically defined transformations
that mimic the innate processing in the human auditory system:

1. AuditoryPrior: Mel-frequency filterbank (mimics cochlear processing)
   - Based on basilar membrane frequency response
   - Logarithmic frequency scale (Weber-Fechner law)
   
2. OnsetDetector: Sudden change detection
   - Based on auditory nerve adaptation
   - Salient for attention and segmentation
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def create_mel_filterbank(
    n_freqs: int,
    n_mels: int,
    sample_rate: int,
    f_min: float = 0.0,
    f_max: Optional[float] = None,
) -> torch.Tensor:
    """
    Create mel-frequency filterbank matrix.
    
    This mimics the frequency response of the basilar membrane in the cochlea,
    which has approximately logarithmic frequency resolution.
    
    Args:
        n_freqs: Number of FFT frequency bins
        n_mels: Number of mel frequency bands
        sample_rate: Audio sample rate in Hz
        f_min: Minimum frequency in Hz
        f_max: Maximum frequency in Hz (default: sample_rate/2)
        
    Returns:
        Filterbank matrix [n_mels, n_freqs]
    """
    f_max = f_max or sample_rate / 2
    
    # Convert Hz to mel scale
    def hz_to_mel(f: float) -> float:
        return 2595.0 * math.log10(1.0 + f / 700.0)
    
    def mel_to_hz(m: float) -> float:
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    
    # Create mel points
    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max)
    mel_points = torch.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = torch.tensor([mel_to_hz(m.item()) for m in mel_points])
    
    # Convert to FFT bin indices
    bin_points = torch.floor((n_freqs + 1) * hz_points / sample_rate).long()
    
    # Create filterbank
    filterbank = torch.zeros(n_mels, n_freqs)
    
    for i in range(n_mels):
        left = bin_points[i]
        center = bin_points[i + 1]
        right = bin_points[i + 2]
        
        # Rising slope
        for j in range(left, center):
            if center != left:
                filterbank[i, j] = (j - left) / (center - left)
        
        # Falling slope
        for j in range(center, right):
            if right != center:
                filterbank[i, j] = (right - j) / (right - center)
    
    return filterbank


class AuditoryPrior(nn.Module):
    """
    Innate auditory processing (cochlear-like).
    
    This transforms raw audio waveforms into mel-spectrograms, which:
    - Use a mel frequency scale (logarithmic, matching human perception)
    - Apply log compression (Weber-Fechner law - perception is logarithmic)
    
    This is the "firmware" of hearing - it exists before learning.
    """
    
    def __init__(
        self, 
        sample_rate: int = 16000, 
        n_mels: int = 80, 
        n_fft: int = 400,
        hop_length: Optional[int] = None,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
    ) -> None:
        """
        Initialize auditory prior.
        
        Args:
            sample_rate: Audio sample rate in Hz
            n_mels: Number of mel frequency bands
            n_fft: FFT window size
            hop_length: Hop between FFT windows (default: n_fft // 4)
            f_min: Minimum frequency for mel filterbank
            f_max: Maximum frequency for mel filterbank
        """
        super().__init__()
        
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length or n_fft // 4
        self.f_min = f_min
        self.f_max = f_max or sample_rate / 2
        
        # Number of frequency bins from FFT
        n_freqs = n_fft // 2 + 1
        
        # Create mel filterbank (fixed, not learned)
        mel_basis = create_mel_filterbank(
            n_freqs=n_freqs,
            n_mels=n_mels,
            sample_rate=sample_rate,
            f_min=f_min,
            f_max=self.f_max,
        )
        self.register_buffer('mel_basis', mel_basis)
        
        # Hann window for STFT
        window = torch.hann_window(n_fft)
        self.register_buffer('window', window)
        
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Transform waveform to log-mel spectrogram.
        
        Args:
            waveform: Raw audio [B, T] where T is number of samples
            
        Returns:
            Log-mel spectrogram [B, n_mels, T'] where T' is time frames
        """
        B = waveform.shape[0]
        
        # Ensure waveform is 2D
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        # Compute STFT
        # Output shape: [B, n_fft//2 + 1, T', 2] for real/imag or [B, n_fft//2 + 1, T'] complex
        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            center=True,
            pad_mode='reflect',
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        
        # Power spectrogram
        power_spec = stft.abs() ** 2  # [B, n_freqs, T']
        
        # Apply mel filterbank
        # mel_basis: [n_mels, n_freqs]
        # power_spec: [B, n_freqs, T']
        mel_spec = torch.matmul(self.mel_basis, power_spec)  # [B, n_mels, T']
        
        # Log compression (Weber-Fechner law)
        # Add small epsilon for numerical stability
        log_mel = torch.log(mel_spec + 1e-9)
        
        return log_mel
    
    def get_time_frames(self, n_samples: int) -> int:
        """Calculate number of time frames for given number of samples."""
        return (n_samples - self.n_fft) // self.hop_length + 1


class OnsetDetector(nn.Module):
    """
    Onset detection prior.
    
    Detects sudden changes in the audio signal, which are perceptually
    salient and important for attention and segmentation.
    
    This is based on the observation that the auditory system is
    particularly sensitive to transients and changes.
    """
    
    def __init__(
        self,
        n_mels: int = 80,
        threshold: float = 0.0,
    ) -> None:
        """
        Initialize onset detector.
        
        Args:
            n_mels: Number of mel bands (for mel spectrogram input)
            threshold: Minimum onset strength (0 = keep all)
        """
        super().__init__()
        self.n_mels = n_mels
        self.threshold = threshold
        
    def forward(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """
        Detect onsets in mel spectrogram.
        
        Args:
            mel_spec: Mel spectrogram [B, n_mels, T]
            
        Returns:
            Onset strength envelope [B, 1, T]
        """
        # Compute first-order difference (spectral flux)
        # Only keep positive changes (onsets, not offsets)
        diff = mel_spec[:, :, 1:] - mel_spec[:, :, :-1]
        positive_diff = F.relu(diff)
        
        # Sum across frequency bands
        onset_strength = positive_diff.sum(dim=1, keepdim=True)  # [B, 1, T-1]
        
        # Pad to match original length
        onset_strength = F.pad(onset_strength, (1, 0), mode='constant', value=0)
        
        # Normalize
        max_val = onset_strength.max(dim=-1, keepdim=True)[0]
        onset_strength = onset_strength / (max_val + 1e-8)
        
        # Apply threshold
        if self.threshold > 0:
            onset_strength = F.relu(onset_strength - self.threshold)
        
        return onset_strength


class SpectralContrastPrior(nn.Module):
    """
    Spectral contrast prior.
    
    Computes the difference between spectral peaks and valleys,
    which is useful for distinguishing harmonic vs. noise content.
    """
    
    def __init__(
        self,
        n_bands: int = 6,
        fmin: float = 200.0,
        quantile: float = 0.02,
    ) -> None:
        """
        Initialize spectral contrast.
        
        Args:
            n_bands: Number of frequency bands for contrast
            fmin: Minimum frequency
            quantile: Fraction of bins for peak/valley estimation
        """
        super().__init__()
        self.n_bands = n_bands
        self.fmin = fmin
        self.quantile = quantile
        
    def forward(self, power_spec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute spectral contrast.
        
        Args:
            power_spec: Power spectrogram [B, n_freqs, T]
            
        Returns:
            Tuple of (peaks [B, n_bands, T], valleys [B, n_bands, T])
        """
        B, n_freqs, T = power_spec.shape
        
        # Divide spectrum into bands (octave-based)
        band_size = n_freqs // self.n_bands
        
        peaks = []
        valleys = []
        
        for i in range(self.n_bands):
            start = i * band_size
            end = (i + 1) * band_size if i < self.n_bands - 1 else n_freqs
            
            band = power_spec[:, start:end, :]  # [B, band_size, T]
            
            # Sort values in each band
            sorted_band, _ = torch.sort(band, dim=1)
            
            # Get top quantile (peaks) and bottom quantile (valleys)
            k = max(1, int(band.shape[1] * self.quantile))
            
            peak = sorted_band[:, -k:, :].mean(dim=1, keepdim=True)  # [B, 1, T]
            valley = sorted_band[:, :k, :].mean(dim=1, keepdim=True)  # [B, 1, T]
            
            peaks.append(peak)
            valleys.append(valley)
        
        peaks = torch.cat(peaks, dim=1)  # [B, n_bands, T]
        valleys = torch.cat(valleys, dim=1)  # [B, n_bands, T]
        
        return peaks, valleys


class PitchPrior(nn.Module):
    """
    Pitch detection prior using autocorrelation.
    
    Estimates fundamental frequency, which is crucial for
    speech and music perception.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        fmin: float = 50.0,
        fmax: float = 500.0,
        frame_length: int = 1024,
        hop_length: int = 256,
    ) -> None:
        """
        Initialize pitch detector.
        
        Args:
            sample_rate: Audio sample rate
            fmin: Minimum detectable pitch (Hz)
            fmax: Maximum detectable pitch (Hz)
            frame_length: Analysis frame length in samples
            hop_length: Hop between frames
        """
        super().__init__()
        self.sample_rate = sample_rate
        self.fmin = fmin
        self.fmax = fmax
        self.frame_length = frame_length
        self.hop_length = hop_length
        
        # Lag range for autocorrelation
        self.lag_min = int(sample_rate / fmax)
        self.lag_max = int(sample_rate / fmin)
        
    def forward(self, waveform: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate pitch using autocorrelation.
        
        Args:
            waveform: Audio waveform [B, T]
            
        Returns:
            Tuple of (pitch_hz [B, T'], confidence [B, T'])
        """
        B, T = waveform.shape
        
        # Frame the signal
        n_frames = (T - self.frame_length) // self.hop_length + 1
        
        pitches = []
        confidences = []
        
        for frame_idx in range(n_frames):
            start = frame_idx * self.hop_length
            frame = waveform[:, start:start + self.frame_length]  # [B, frame_length]
            
            # Compute autocorrelation via FFT
            fft = torch.fft.rfft(frame, n=2 * self.frame_length)
            power = fft.abs() ** 2
            autocorr = torch.fft.irfft(power)[:, :self.frame_length]
            
            # Normalize
            autocorr = autocorr / (autocorr[:, 0:1] + 1e-8)
            
            # Find peak in valid lag range
            valid_autocorr = autocorr[:, self.lag_min:self.lag_max]
            
            if valid_autocorr.shape[1] > 0:
                peak_val, peak_idx = valid_autocorr.max(dim=1)
                peak_lag = peak_idx + self.lag_min
                
                # Convert lag to Hz
                pitch_hz = self.sample_rate / peak_lag.float()
                confidence = peak_val
            else:
                pitch_hz = torch.zeros(B, device=waveform.device)
                confidence = torch.zeros(B, device=waveform.device)
            
            pitches.append(pitch_hz.unsqueeze(1))
            confidences.append(confidence.unsqueeze(1))
        
        pitches = torch.cat(pitches, dim=1)  # [B, n_frames]
        confidences = torch.cat(confidences, dim=1)  # [B, n_frames]
        
        return pitches, confidences
