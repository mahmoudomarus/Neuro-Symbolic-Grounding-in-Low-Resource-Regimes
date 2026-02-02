"""
Enhanced Video Augmentation for NSCA.

Includes:
- RandAugment: Learned augmentation policies
- MixUp/CutMix: Regularization for better generalization
- Temporal augmentation: Frame dropping, speed perturbation
- Physics-aware mode: Disables gravity-inconsistent transforms
"""

import random
from typing import Tuple, Optional, List, Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


@dataclass
class VideoAugConfig:
    """Configuration for video augmentation."""
    # RandAugment parameters
    rand_augment_n: int = 2  # Number of augmentations to apply
    rand_augment_m: int = 9  # Magnitude (0-10)
    
    # MixUp/CutMix
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0
    mix_prob: float = 0.5
    
    # Temporal
    temporal_dropout: float = 0.1  # Probability of dropping frames
    speed_range: Tuple[float, float] = (0.8, 1.2)
    
    # Physics-aware
    physics_aware: bool = True  # Disable vertical flips for physics tasks
    
    # Basic augmentation probabilities
    horizontal_flip_p: float = 0.5
    brightness_p: float = 0.5
    brightness_range: Tuple[float, float] = (0.8, 1.2)
    contrast_p: float = 0.5
    contrast_range: Tuple[float, float] = (0.8, 1.2)
    crop_p: float = 0.5
    crop_scale: Tuple[float, float] = (0.85, 1.0)
    
    # Color jitter
    color_jitter_p: float = 0.8
    color_jitter_brightness: float = 0.4
    color_jitter_contrast: float = 0.4
    color_jitter_saturation: float = 0.4
    color_jitter_hue: float = 0.1
    
    # Grayscale
    grayscale_p: float = 0.2


class RandAugment:
    """
    RandAugment: Practical automated data augmentation.
    
    Randomly selects N augmentations from a pool and applies them
    with magnitude M.
    """
    
    def __init__(self, n: int = 2, m: int = 9, physics_aware: bool = True):
        """
        Args:
            n: Number of augmentations to apply
            m: Magnitude of augmentations (0-10)
            physics_aware: If True, excludes vertical flips
        """
        self.n = n
        self.m = m
        self.physics_aware = physics_aware
        
        # Define augmentation pool
        self.augmentations = [
            ('identity', self._identity),
            ('autocontrast', self._autocontrast),
            ('equalize', self._equalize),
            ('rotate', self._rotate),
            ('solarize', self._solarize),
            ('posterize', self._posterize),
            ('contrast', self._contrast),
            ('brightness', self._brightness),
            ('sharpness', self._sharpness),
            ('shear_x', self._shear_x),
            ('shear_y', self._shear_y),
            ('translate_x', self._translate_x),
            ('translate_y', self._translate_y),
        ]
        
        if not physics_aware:
            self.augmentations.append(('flip_vertical', self._flip_vertical))
    
    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Apply random augmentations to video frames.
        
        Args:
            frames: [T, C, H, W] video frames
            
        Returns:
            Augmented frames [T, C, H, W]
        """
        # Sample N augmentations
        ops = random.sample(self.augmentations, self.n)
        
        for name, op in ops:
            frames = op(frames, self.m)
        
        return frames
    
    def _identity(self, x: torch.Tensor, m: int) -> torch.Tensor:
        return x
    
    def _autocontrast(self, x: torch.Tensor, m: int) -> torch.Tensor:
        """Apply autocontrast."""
        # Per-channel min-max normalization
        for t in range(x.shape[0]):
            for c in range(x.shape[1]):
                channel = x[t, c]
                min_val = channel.min()
                max_val = channel.max()
                if max_val - min_val > 1e-5:
                    x[t, c] = (channel - min_val) / (max_val - min_val)
        return x
    
    def _equalize(self, x: torch.Tensor, m: int) -> torch.Tensor:
        """Simple histogram equalization approximation."""
        return torch.clamp(x * 1.1, -2.5, 2.5)
    
    def _rotate(self, x: torch.Tensor, m: int) -> torch.Tensor:
        """Rotate frames by small angle."""
        angle = (m / 10.0) * 30 * (2 * random.random() - 1)  # ±30 degrees max
        angle_rad = angle * np.pi / 180
        
        T, C, H, W = x.shape
        
        # Create rotation matrix
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        theta = torch.tensor([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0]
        ], dtype=x.dtype, device=x.device).unsqueeze(0).expand(T, -1, -1)
        
        grid = F.affine_grid(theta, [T, C, H, W], align_corners=False)
        return F.grid_sample(x, grid, align_corners=False, mode='bilinear', padding_mode='reflection')
    
    def _solarize(self, x: torch.Tensor, m: int) -> torch.Tensor:
        """Invert pixels above threshold."""
        threshold = 1.0 - (m / 10.0)
        mask = x > threshold
        x = x.clone()
        x[mask] = 2 * threshold - x[mask]
        return x
    
    def _posterize(self, x: torch.Tensor, m: int) -> torch.Tensor:
        """Reduce bits per channel."""
        bits = 8 - int((m / 10.0) * 4)
        bits = max(1, bits)
        # Quantize
        levels = 2 ** bits
        return torch.round(x * levels) / levels
    
    def _contrast(self, x: torch.Tensor, m: int) -> torch.Tensor:
        """Adjust contrast."""
        factor = 1.0 + (m / 10.0) * (2 * random.random() - 1)
        mean = x.mean(dim=(2, 3), keepdim=True)
        return torch.clamp((x - mean) * factor + mean, -2.5, 2.5)
    
    def _brightness(self, x: torch.Tensor, m: int) -> torch.Tensor:
        """Adjust brightness."""
        factor = 1.0 + (m / 10.0) * 0.5 * (2 * random.random() - 1)
        return torch.clamp(x * factor, -2.5, 2.5)
    
    def _sharpness(self, x: torch.Tensor, m: int) -> torch.Tensor:
        """Simple sharpening via high-pass filter."""
        # Create simple sharpening kernel
        kernel = torch.tensor([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
        
        factor = (m / 10.0) * 0.5
        
        T, C, H, W = x.shape
        x_flat = x.contiguous().reshape(T * C, 1, H, W)
        sharpened = F.conv2d(x_flat, kernel, padding=1)
        sharpened = sharpened.reshape(T, C, H, W)
        
        return torch.clamp(x + factor * (sharpened - x), -2.5, 2.5)
    
    def _shear_x(self, x: torch.Tensor, m: int) -> torch.Tensor:
        """Shear along x-axis."""
        shear = (m / 10.0) * 0.3 * (2 * random.random() - 1)
        T, C, H, W = x.shape
        
        theta = torch.tensor([
            [1, shear, 0],
            [0, 1, 0]
        ], dtype=x.dtype, device=x.device).unsqueeze(0).expand(T, -1, -1)
        
        grid = F.affine_grid(theta, [T, C, H, W], align_corners=False)
        return F.grid_sample(x, grid, align_corners=False, mode='bilinear', padding_mode='reflection')
    
    def _shear_y(self, x: torch.Tensor, m: int) -> torch.Tensor:
        """Shear along y-axis."""
        shear = (m / 10.0) * 0.3 * (2 * random.random() - 1)
        T, C, H, W = x.shape
        
        theta = torch.tensor([
            [1, 0, 0],
            [shear, 1, 0]
        ], dtype=x.dtype, device=x.device).unsqueeze(0).expand(T, -1, -1)
        
        grid = F.affine_grid(theta, [T, C, H, W], align_corners=False)
        return F.grid_sample(x, grid, align_corners=False, mode='bilinear', padding_mode='reflection')
    
    def _translate_x(self, x: torch.Tensor, m: int) -> torch.Tensor:
        """Translate along x-axis."""
        trans = (m / 10.0) * 0.3 * (2 * random.random() - 1)
        T, C, H, W = x.shape
        
        theta = torch.tensor([
            [1, 0, trans],
            [0, 1, 0]
        ], dtype=x.dtype, device=x.device).unsqueeze(0).expand(T, -1, -1)
        
        grid = F.affine_grid(theta, [T, C, H, W], align_corners=False)
        return F.grid_sample(x, grid, align_corners=False, mode='bilinear', padding_mode='reflection')
    
    def _translate_y(self, x: torch.Tensor, m: int) -> torch.Tensor:
        """Translate along y-axis."""
        trans = (m / 10.0) * 0.3 * (2 * random.random() - 1)
        T, C, H, W = x.shape
        
        theta = torch.tensor([
            [1, 0, 0],
            [0, 1, trans]
        ], dtype=x.dtype, device=x.device).unsqueeze(0).expand(T, -1, -1)
        
        grid = F.affine_grid(theta, [T, C, H, W], align_corners=False)
        return F.grid_sample(x, grid, align_corners=False, mode='bilinear', padding_mode='reflection')
    
    def _flip_vertical(self, x: torch.Tensor, m: int) -> torch.Tensor:
        """Flip vertically (only if not physics-aware)."""
        return torch.flip(x, dims=[2])


class MixUp:
    """
    MixUp augmentation for video.
    
    Mixes two samples: x' = λx_i + (1-λ)x_j
    """
    
    def __init__(self, alpha: float = 0.2):
        """
        Args:
            alpha: Beta distribution parameter for mixing coefficient
        """
        self.alpha = alpha
    
    def __call__(
        self,
        frames1: torch.Tensor,
        frames2: torch.Tensor,
        label1: Optional[torch.Tensor] = None,
        label2: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], float]:
        """
        Mix two video samples.
        
        Args:
            frames1: First video [T, C, H, W]
            frames2: Second video [T, C, H, W]
            label1: Optional label for first sample
            label2: Optional label for second sample
            
        Returns:
            Mixed frames, mixed labels (if provided), mixing coefficient
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0
        
        mixed_frames = lam * frames1 + (1 - lam) * frames2
        
        if label1 is not None and label2 is not None:
            mixed_labels = lam * label1 + (1 - lam) * label2
            return mixed_frames, mixed_labels, lam
        
        return mixed_frames, None, lam


class CutMix:
    """
    CutMix augmentation for video.
    
    Cuts a region from one sample and pastes into another.
    """
    
    def __init__(self, alpha: float = 1.0):
        """
        Args:
            alpha: Beta distribution parameter for cut ratio
        """
        self.alpha = alpha
    
    def __call__(
        self,
        frames1: torch.Tensor,
        frames2: torch.Tensor,
        label1: Optional[torch.Tensor] = None,
        label2: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], float]:
        """
        Apply CutMix to two video samples.
        
        Args:
            frames1: First video [T, C, H, W]
            frames2: Second video [T, C, H, W]
            label1: Optional label for first sample
            label2: Optional label for second sample
            
        Returns:
            Mixed frames, mixed labels (if provided), mixing coefficient
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0
        
        T, C, H, W = frames1.shape
        
        # Get cut coordinates
        cut_rat = np.sqrt(1 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        # Uniform center
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        # Apply cut
        mixed_frames = frames1.clone()
        mixed_frames[:, :, bby1:bby2, bbx1:bbx2] = frames2[:, :, bby1:bby2, bbx1:bbx2]
        
        # Adjust lambda to actual cut ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        
        if label1 is not None and label2 is not None:
            mixed_labels = lam * label1 + (1 - lam) * label2
            return mixed_frames, mixed_labels, lam
        
        return mixed_frames, None, lam


class TemporalAugmentation:
    """
    Temporal augmentations for video.
    
    Includes:
    - Frame dropout: Randomly drop frames
    - Speed perturbation: Stretch/compress time
    - Temporal shift: Random offset in time
    """
    
    def __init__(
        self,
        dropout_p: float = 0.1,
        speed_range: Tuple[float, float] = (0.8, 1.2),
        shift_range: Tuple[int, int] = (-2, 2),
    ):
        """
        Args:
            dropout_p: Probability of dropping each frame
            speed_range: Range for speed factor (< 1 slower, > 1 faster)
            shift_range: Range for temporal shift in frames
        """
        self.dropout_p = dropout_p
        self.speed_range = speed_range
        self.shift_range = shift_range
    
    def frame_dropout(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Randomly drop frames and interpolate.
        
        Args:
            frames: [T, C, H, W]
            
        Returns:
            Frames with some dropped and interpolated
        """
        T, C, H, W = frames.shape
        
        if T <= 2:
            return frames
        
        # Create dropout mask (keep at least first and last)
        keep_mask = torch.rand(T) > self.dropout_p
        keep_mask[0] = True
        keep_mask[-1] = True
        
        # If all kept, return as-is
        if keep_mask.all():
            return frames
        
        # Get kept frames
        kept_indices = torch.where(keep_mask)[0]
        kept_frames = frames[kept_indices]
        
        # Interpolate to original length
        # Reshape for interpolation: [1, C*T_kept, H, W] -> interpolate -> [1, C*T, H, W]
        kept_frames = kept_frames.permute(1, 0, 2, 3)  # [C, T_kept, H, W]
        kept_frames = kept_frames.unsqueeze(0)  # [1, C, T_kept, H, W]
        
        # 3D interpolation
        interpolated = F.interpolate(
            kept_frames,
            size=(T, H, W),
            mode='trilinear',
            align_corners=False
        )
        
        return interpolated.squeeze(0).permute(1, 0, 2, 3)  # [T, C, H, W]
    
    def speed_perturb(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Randomly change video speed.
        
        Args:
            frames: [T, C, H, W]
            
        Returns:
            Speed-adjusted frames (same length via interpolation)
        """
        T, C, H, W = frames.shape
        
        speed = random.uniform(*self.speed_range)
        new_T = int(T / speed)
        new_T = max(2, new_T)  # At least 2 frames
        
        # Resample temporally
        frames = frames.permute(1, 0, 2, 3).unsqueeze(0)  # [1, C, T, H, W]
        
        resampled = F.interpolate(
            frames,
            size=(new_T, H, W),
            mode='trilinear',
            align_corners=False
        )
        
        # Resize back to original T
        resampled = F.interpolate(
            resampled,
            size=(T, H, W),
            mode='trilinear',
            align_corners=False
        )
        
        return resampled.squeeze(0).permute(1, 0, 2, 3)  # [T, C, H, W]
    
    def temporal_shift(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Apply temporal shift (circular).
        
        Args:
            frames: [T, C, H, W]
            
        Returns:
            Temporally shifted frames
        """
        shift = random.randint(*self.shift_range)
        return torch.roll(frames, shifts=shift, dims=0)
    
    def __call__(
        self,
        frames: torch.Tensor,
        dropout: bool = True,
        speed: bool = True,
        shift: bool = False,
    ) -> torch.Tensor:
        """
        Apply temporal augmentations.
        
        Args:
            frames: [T, C, H, W]
            dropout: Apply frame dropout
            speed: Apply speed perturbation
            shift: Apply temporal shift
            
        Returns:
            Augmented frames
        """
        if dropout and random.random() < 0.5:
            frames = self.frame_dropout(frames)
        
        if speed and random.random() < 0.5:
            frames = self.speed_perturb(frames)
        
        if shift and random.random() < 0.5:
            frames = self.temporal_shift(frames)
        
        return frames


class EnhancedVideoAugmentation:
    """
    Enhanced video augmentation combining all techniques.
    
    Includes:
    - RandAugment for learned policies
    - MixUp/CutMix for regularization
    - Temporal augmentation
    - Physics-aware mode
    """
    
    def __init__(
        self,
        config: Optional[VideoAugConfig] = None,
        physics_aware: bool = True,
        p: float = 0.5,
    ):
        """
        Args:
            config: Augmentation configuration
            physics_aware: Disable gravity-inconsistent transforms
            p: Probability of applying augmentation
        """
        self.config = config or VideoAugConfig()
        self.physics_aware = physics_aware
        self.p = p
        
        # Initialize sub-augmentations
        self.rand_augment = RandAugment(
            n=self.config.rand_augment_n,
            m=self.config.rand_augment_m,
            physics_aware=physics_aware,
        )
        self.mixup = MixUp(alpha=self.config.mixup_alpha)
        self.cutmix = CutMix(alpha=self.config.cutmix_alpha)
        self.temporal = TemporalAugmentation(
            dropout_p=self.config.temporal_dropout,
            speed_range=self.config.speed_range,
        )
    
    def __call__(
        self,
        frames: torch.Tensor,
        frames2: Optional[torch.Tensor] = None,
        label: Optional[torch.Tensor] = None,
        label2: Optional[torch.Tensor] = None,
        use_rand_augment: bool = True,
        use_temporal: bool = True,
        use_mix: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], float]:
        """
        Apply enhanced video augmentation.
        
        Args:
            frames: Primary video [T, C, H, W] or batched [B, T, C, H, W]
            frames2: Optional second video for mixing
            label: Optional label for primary
            label2: Optional label for second
            use_rand_augment: Apply RandAugment
            use_temporal: Apply temporal augmentation
            use_mix: Apply MixUp or CutMix (requires frames2)
            
        Returns:
            Augmented frames, optional mixed labels, mixing coefficient
        """
        if random.random() > self.p:
            return frames, label, 1.0
        
        # Handle batched input [B, T, C, H, W]
        batched = frames.dim() == 5
        if batched:
            # Process each sample in batch separately
            B = frames.shape[0]
            augmented = []
            for i in range(B):
                aug_frame, _, _ = self._augment_single(
                    frames[i], None, None, None,
                    use_rand_augment, use_temporal, False  # No mixing for batched
                )
                augmented.append(aug_frame)
            return torch.stack(augmented), label, 1.0
        
        return self._augment_single(
            frames, frames2, label, label2,
            use_rand_augment, use_temporal, use_mix
        )
    
    def _augment_single(
        self,
        frames: torch.Tensor,
        frames2: Optional[torch.Tensor],
        label: Optional[torch.Tensor],
        label2: Optional[torch.Tensor],
        use_rand_augment: bool,
        use_temporal: bool,
        use_mix: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], float]:
        """Apply augmentation to a single video [T, C, H, W]."""
        mix_lambda = 1.0
        mixed_label = label
        
        # Apply basic augmentations first
        frames = self._basic_augment(frames)
        
        # RandAugment
        if use_rand_augment:
            frames = self.rand_augment(frames)
        
        # Temporal augmentation
        if use_temporal:
            frames = self.temporal(frames)
        
        # MixUp or CutMix (if second sample provided)
        if use_mix and frames2 is not None and random.random() < self.config.mix_prob:
            # Also augment second sample
            frames2 = self._basic_augment(frames2)
            if use_rand_augment:
                frames2 = self.rand_augment(frames2)
            
            if random.random() < 0.5:
                frames, mixed_label, mix_lambda = self.mixup(
                    frames, frames2, label, label2
                )
            else:
                frames, mixed_label, mix_lambda = self.cutmix(
                    frames, frames2, label, label2
                )
        
        return frames, mixed_label, mix_lambda
    
    def _basic_augment(self, frames: torch.Tensor) -> torch.Tensor:
        """Apply basic augmentations to [T, C, H, W] tensor."""
        T, C, H, W = frames.shape
        
        # Horizontal flip (always allowed)
        if random.random() < self.config.horizontal_flip_p:
            frames = torch.flip(frames, dims=[3])
        
        # Brightness
        if random.random() < self.config.brightness_p:
            factor = random.uniform(*self.config.brightness_range)
            frames = frames * factor
            frames = torch.clamp(frames, -2.5, 2.5)
        
        # Contrast
        if random.random() < self.config.contrast_p:
            factor = random.uniform(*self.config.contrast_range)
            mean = frames.mean(dim=(2, 3), keepdim=True)
            frames = (frames - mean) * factor + mean
            frames = torch.clamp(frames, -2.5, 2.5)
        
        # Random crop and resize
        if random.random() < self.config.crop_p:
            crop_scale = random.uniform(*self.config.crop_scale)
            crop_size = int(H * crop_scale)
            top = random.randint(0, H - crop_size)
            left = random.randint(0, W - crop_size)
            frames = frames[:, :, top:top+crop_size, left:left+crop_size]
            frames = F.interpolate(frames, size=(H, W), mode='bilinear', align_corners=False)
        
        # Color jitter (simple version)
        if random.random() < self.config.color_jitter_p:
            # Random saturation adjustment
            if frames.shape[1] == 3:  # RGB only
                gray = frames.mean(dim=1, keepdim=True)
                sat_factor = random.uniform(
                    1 - self.config.color_jitter_saturation,
                    1 + self.config.color_jitter_saturation
                )
                frames = sat_factor * frames + (1 - sat_factor) * gray
                frames = torch.clamp(frames, -2.5, 2.5)
        
        # Random grayscale
        if random.random() < self.config.grayscale_p:
            if frames.shape[1] == 3:
                gray = frames.mean(dim=1, keepdim=True)
                frames = gray.expand_as(frames)
        
        return frames
