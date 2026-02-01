"""
Audio encoder with innate priors.

This encoder applies cochlear-like processing BEFORE any learned features:

1. Mel spectrogram (mimics basilar membrane frequency response)
2. Log compression (Weber-Fechner law - perception is logarithmic)
3. Onset detection (transient salience)

Then learned convolutional features extract higher-level representations.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.priors.audio_prior import AuditoryPrior, OnsetDetector


@dataclass
class AudioEncoderConfig:
    """Configuration for audio encoder."""
    sample_rate: int = 16000
    n_mels: int = 80
    n_fft: int = 400
    hop_length: int = 160
    latent_dim: int = 256
    output_dim: int = 512
    use_onset_prior: bool = True
    max_audio_length: int = 16000 * 5  # 5 seconds


class AudioEncoderWithPriors(nn.Module):
    """
    Audio encoder that applies cochlear-like processing before learning.
    
    Pipeline:
    1. Mel spectrogram (innate - mimics cochlea)
    2. Log compression (innate - Weber-Fechner law)
    3. Onset detection (innate - transient salience)
    4. Learned encoder -> Abstract features
    
    This ensures the encoder has innate auditory structure.
    """
    
    def __init__(self, config: AudioEncoderConfig) -> None:
        super().__init__()
        
        self.config = config
        
        # ===== INNATE PRIORS (not trained) =====
        
        self.auditory_prior = AuditoryPrior(
            sample_rate=config.sample_rate,
            n_mels=config.n_mels,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
        )
        
        if config.use_onset_prior:
            self.onset_prior = OnsetDetector(n_mels=config.n_mels)
        else:
            self.onset_prior = None
        
        # Input channels: 1 (mel) + 1 (onset) if using onset
        in_channels = 2 if config.use_onset_prior else 1
        
        # ===== LEARNED ENCODER =====
        
        # CNN over spectrogram
        self.conv_encoder = nn.Sequential(
            # [B, in_channels, n_mels, T]
            nn.Conv2d(in_channels, 32, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            
            nn.Conv2d(128, config.latent_dim, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(config.latent_dim),
            nn.GELU(),
        )
        
        # Projection to output dimension
        self.project = nn.Linear(config.latent_dim, config.output_dim)
        
        self.out_channels = config.output_dim
    
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Encode waveform to latent representation.
        
        Args:
            waveform: Raw audio [B, T] where T is number of samples
            
        Returns:
            Audio features [B, output_dim]
        """
        # Ensure waveform is 2D
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        B = waveform.shape[0]
        
        # Apply innate mel processing
        mel = self.auditory_prior(waveform)  # [B, n_mels, T']
        
        # Stack with onset if using
        if self.onset_prior is not None:
            onset = self.onset_prior(mel)  # [B, 1, T']
            # Expand onset to match mel's mel dimension
            onset_expanded = onset.expand(-1, self.config.n_mels, -1)  # [B, n_mels, T']
            mel_input = torch.stack([mel, onset_expanded], dim=1)  # [B, 2, n_mels, T']
        else:
            mel_input = mel.unsqueeze(1)  # [B, 1, n_mels, T']
        
        # Learned encoding
        features = self.conv_encoder(mel_input)  # [B, latent_dim, H', W']
        
        # Global average pooling
        features = features.mean(dim=(2, 3))  # [B, latent_dim]
        
        # Project to output dim
        features = self.project(features)  # [B, output_dim]
        
        return features
    
    def forward_temporal(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Encode waveform preserving temporal structure.
        
        Args:
            waveform: Raw audio [B, T]
            
        Returns:
            Temporal features [B, T', output_dim]
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        # Apply mel processing
        mel = self.auditory_prior(waveform)  # [B, n_mels, T']
        
        if self.onset_prior is not None:
            onset = self.onset_prior(mel)
            onset_expanded = onset.expand(-1, self.config.n_mels, -1)
            mel_input = torch.stack([mel, onset_expanded], dim=1)
        else:
            mel_input = mel.unsqueeze(1)
        
        # Encode but preserve time dimension
        features = self.conv_encoder(mel_input)  # [B, latent_dim, H', T'']
        
        # Pool over frequency, keep time
        features = features.mean(dim=2)  # [B, latent_dim, T'']
        features = features.permute(0, 2, 1)  # [B, T'', latent_dim]
        
        # Project each time step
        features = self.project(features)  # [B, T'', output_dim]
        
        return features
    
    def get_mel_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Get mel spectrogram without further encoding.
        
        Args:
            waveform: Raw audio [B, T]
            
        Returns:
            Log-mel spectrogram [B, n_mels, T']
        """
        return self.auditory_prior(waveform)


class AudioEncoderLite(nn.Module):
    """
    Lightweight audio encoder for faster inference.
    """
    
    def __init__(self, config: AudioEncoderConfig) -> None:
        super().__init__()
        
        self.config = config
        self.auditory_prior = AuditoryPrior(
            sample_rate=config.sample_rate,
            n_mels=config.n_mels,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
        )
        
        # Simpler encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(config.n_mels, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(128, config.output_dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        
        self.out_channels = config.output_dim
    
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Encode waveform to features."""
        mel = self.auditory_prior(waveform)  # [B, n_mels, T']
        features = self.encoder(mel)  # [B, output_dim, T'']
        return features.mean(dim=2)  # [B, output_dim]
