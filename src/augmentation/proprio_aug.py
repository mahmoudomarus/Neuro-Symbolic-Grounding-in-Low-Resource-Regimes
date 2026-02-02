"""
Proprioception Augmentation for NSCA.

Includes:
- Sensor noise injection: Gaussian noise to simulate sensor uncertainty
- Joint dropout: Randomly zero some joints for robustness
- Temporal jitter: Small random shifts in timing
- Coordinate perturbation: Small random offsets
"""

import random
from typing import Tuple, Optional, List
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


@dataclass
class ProprioAugConfig:
    """Configuration for proprioception augmentation."""
    # Sensor noise
    noise_std: float = 0.01
    noise_type: str = 'gaussian'  # 'gaussian', 'uniform', 'laplace'
    
    # Joint dropout
    joint_dropout_p: float = 0.1
    min_joints_kept: int = 6  # Minimum joints to keep (out of typical 12)
    
    # Temporal jitter
    temporal_jitter_ms: int = 10
    sample_rate: int = 100  # Hz
    
    # Coordinate perturbation
    position_noise: float = 0.005  # meters
    velocity_noise: float = 0.01  # m/s
    acceleration_noise: float = 0.02  # m/s^2
    orientation_noise: float = 0.01  # radians
    
    # Scaling
    scale_range: Tuple[float, float] = (0.95, 1.05)
    
    # Drift simulation
    drift_prob: float = 0.1
    drift_magnitude: float = 0.02


class SensorNoiseAugmentation:
    """
    Adds realistic sensor noise to proprioception data.
    
    Models different types of sensor noise commonly found in
    robotic systems and motion capture.
    """
    
    def __init__(
        self,
        noise_std: float = 0.01,
        noise_type: str = 'gaussian',
        channel_specific: bool = True,
    ):
        """
        Args:
            noise_std: Standard deviation of noise
            noise_type: Type of noise distribution
            channel_specific: Apply different noise levels per channel
        """
        self.noise_std = noise_std
        self.noise_type = noise_type
        self.channel_specific = channel_specific
        
        # Typical noise levels for different proprioceptive channels
        # [position (3), velocity (3), acceleration (3), orientation (3)]
        self.channel_noise_scale = torch.tensor([
            1.0, 1.0, 1.0,  # Position: moderate noise
            1.5, 1.5, 1.5,  # Velocity: higher noise (derivative)
            2.0, 2.0, 2.0,  # Acceleration: highest noise (second derivative)
            0.8, 0.8, 0.8,  # Orientation: lower noise (typically from IMU)
        ])
    
    def __call__(self, proprio: torch.Tensor) -> torch.Tensor:
        """
        Add sensor noise to proprioception data.
        
        Args:
            proprio: [T, D] or [B, T, D] proprioception data
            
        Returns:
            Noisy proprioception data
        """
        if self.noise_type == 'gaussian':
            noise = torch.randn_like(proprio) * self.noise_std
        elif self.noise_type == 'uniform':
            noise = (torch.rand_like(proprio) * 2 - 1) * self.noise_std * np.sqrt(3)
        elif self.noise_type == 'laplace':
            # Laplace distribution for heavier tails
            u = torch.rand_like(proprio) - 0.5
            noise = -self.noise_std * torch.sign(u) * torch.log(1 - 2 * u.abs() + 1e-8)
        else:
            noise = torch.randn_like(proprio) * self.noise_std
        
        # Apply channel-specific scaling
        if self.channel_specific and proprio.shape[-1] == 12:
            scale = self.channel_noise_scale.to(proprio.device)
            if proprio.dim() == 2:
                noise = noise * scale
            else:  # [B, T, D]
                noise = noise * scale.view(1, 1, -1)
        
        return proprio + noise


class JointDropout:
    """
    Randomly zeros out joint values to improve robustness.
    
    Simulates sensor failures and occlusions.
    """
    
    def __init__(
        self,
        dropout_p: float = 0.1,
        min_joints_kept: int = 6,
        replace_with: str = 'zero',  # 'zero', 'mean', 'prev'
    ):
        """
        Args:
            dropout_p: Probability of dropping each joint
            min_joints_kept: Minimum number of joints to keep
            replace_with: How to replace dropped values
        """
        self.dropout_p = dropout_p
        self.min_joints_kept = min_joints_kept
        self.replace_with = replace_with
    
    def __call__(self, proprio: torch.Tensor) -> torch.Tensor:
        """
        Apply joint dropout to proprioception data.
        
        Args:
            proprio: [T, D] or [B, T, D] proprioception data
            
        Returns:
            Dropped-out proprioception data
        """
        D = proprio.shape[-1]
        
        # Generate dropout mask
        if proprio.dim() == 2:
            T, D = proprio.shape
            mask = torch.rand(D, device=proprio.device) > self.dropout_p
        else:
            B, T, D = proprio.shape
            mask = torch.rand(B, D, device=proprio.device) > self.dropout_p
        
        # Ensure minimum joints kept
        num_kept = mask.sum(dim=-1)
        if isinstance(num_kept, torch.Tensor) and num_kept.dim() > 0:
            # Batch case
            for i in range(len(num_kept)):
                while num_kept[i] < self.min_joints_kept:
                    idx = random.randint(0, D - 1)
                    if not mask[i, idx]:
                        mask[i, idx] = True
                        num_kept[i] += 1
        else:
            while num_kept < self.min_joints_kept:
                idx = random.randint(0, D - 1)
                if not mask[idx]:
                    mask[idx] = True
                    num_kept += 1
        
        # Apply dropout
        proprio = proprio.clone()
        
        if self.replace_with == 'zero':
            if proprio.dim() == 2:
                proprio[:, ~mask] = 0
            else:
                mask_expanded = mask.unsqueeze(1).expand(-1, T, -1)
                proprio[~mask_expanded] = 0
        
        elif self.replace_with == 'mean':
            mean_val = proprio.mean()
            if proprio.dim() == 2:
                proprio[:, ~mask] = mean_val
            else:
                mask_expanded = mask.unsqueeze(1).expand(-1, T, -1)
                proprio[~mask_expanded] = mean_val
        
        elif self.replace_with == 'prev':
            # Replace with previous timestep value (forward fill)
            if proprio.dim() == 2:
                for d in range(D):
                    if not mask[d]:
                        for t in range(1, T):
                            proprio[t, d] = proprio[t-1, d]
            else:
                for b in range(B):
                    for d in range(D):
                        if not mask[b, d]:
                            for t in range(1, T):
                                proprio[b, t, d] = proprio[b, t-1, d]
        
        return proprio


class TemporalJitter:
    """
    Applies temporal jitter to simulate timing variations.
    
    Models clock drift and synchronization issues.
    """
    
    def __init__(
        self,
        jitter_ms: int = 10,
        sample_rate: int = 100,  # Hz
    ):
        """
        Args:
            jitter_ms: Maximum jitter in milliseconds
            sample_rate: Sample rate in Hz
        """
        self.jitter_ms = jitter_ms
        self.sample_rate = sample_rate
        self.max_shift_samples = int(jitter_ms * sample_rate / 1000)
    
    def __call__(self, proprio: torch.Tensor) -> torch.Tensor:
        """
        Apply temporal jitter to proprioception data.
        
        Args:
            proprio: [T, D] or [B, T, D] proprioception data
            
        Returns:
            Temporally jittered proprioception data
        """
        if self.max_shift_samples == 0:
            return proprio
        
        # Random shift per sample
        shift = random.randint(-self.max_shift_samples, self.max_shift_samples)
        
        if shift == 0:
            return proprio
        
        # Apply circular shift
        return torch.roll(proprio, shifts=shift, dims=-2 if proprio.dim() == 3 else 0)
    
    def non_uniform_jitter(self, proprio: torch.Tensor) -> torch.Tensor:
        """
        Apply non-uniform temporal jitter (different per timestep).
        
        Args:
            proprio: [T, D] or [B, T, D]
            
        Returns:
            Jittered proprioception data
        """
        if proprio.dim() == 2:
            T, D = proprio.shape
            batch = False
        else:
            B, T, D = proprio.shape
            batch = True
        
        # Generate random offsets per timestep
        offsets = torch.randint(
            -self.max_shift_samples,
            self.max_shift_samples + 1,
            (T,),
            device=proprio.device
        )
        
        # Create new indices
        indices = torch.arange(T, device=proprio.device) + offsets
        indices = indices.clamp(0, T - 1)
        
        # Gather
        if batch:
            indices = indices.view(1, T, 1).expand(B, -1, D)
            return torch.gather(proprio, 1, indices)
        else:
            indices = indices.view(T, 1).expand(-1, D)
            return torch.gather(proprio, 0, indices)


class CoordinatePerturbation:
    """
    Applies coordinate-specific perturbations to proprioception.
    
    Models calibration errors and measurement uncertainty.
    """
    
    def __init__(
        self,
        position_noise: float = 0.005,
        velocity_noise: float = 0.01,
        acceleration_noise: float = 0.02,
        orientation_noise: float = 0.01,
    ):
        """
        Args:
            position_noise: Noise for position (meters)
            velocity_noise: Noise for velocity (m/s)
            acceleration_noise: Noise for acceleration (m/s^2)
            orientation_noise: Noise for orientation (radians)
        """
        self.noise_levels = torch.tensor([
            position_noise, position_noise, position_noise,
            velocity_noise, velocity_noise, velocity_noise,
            acceleration_noise, acceleration_noise, acceleration_noise,
            orientation_noise, orientation_noise, orientation_noise,
        ])
    
    def __call__(self, proprio: torch.Tensor) -> torch.Tensor:
        """
        Apply coordinate-specific perturbations.
        
        Args:
            proprio: [T, D] or [B, T, D] with D=12
            
        Returns:
            Perturbed proprioception data
        """
        D = proprio.shape[-1]
        
        if D != 12:
            # Fall back to uniform noise
            return proprio + torch.randn_like(proprio) * 0.01
        
        noise_levels = self.noise_levels.to(proprio.device)
        
        # Generate noise
        noise = torch.randn_like(proprio)
        
        # Scale by channel-specific noise levels
        if proprio.dim() == 2:
            noise = noise * noise_levels
        else:
            noise = noise * noise_levels.view(1, 1, -1)
        
        return proprio + noise


class DriftSimulation:
    """
    Simulates sensor drift over time.
    
    Models gradual calibration drift common in sensors.
    """
    
    def __init__(
        self,
        drift_prob: float = 0.1,
        drift_magnitude: float = 0.02,
        drift_type: str = 'linear',  # 'linear', 'random_walk'
    ):
        """
        Args:
            drift_prob: Probability of applying drift
            drift_magnitude: Maximum drift magnitude
            drift_type: Type of drift pattern
        """
        self.drift_prob = drift_prob
        self.drift_magnitude = drift_magnitude
        self.drift_type = drift_type
    
    def __call__(self, proprio: torch.Tensor) -> torch.Tensor:
        """
        Apply drift simulation.
        
        Args:
            proprio: [T, D] or [B, T, D]
            
        Returns:
            Drifted proprioception data
        """
        if random.random() > self.drift_prob:
            return proprio
        
        if proprio.dim() == 2:
            T, D = proprio.shape
            batch = False
        else:
            B, T, D = proprio.shape
            batch = True
        
        # Generate drift pattern
        if self.drift_type == 'linear':
            # Linear drift
            drift = torch.linspace(0, 1, T, device=proprio.device)
            drift = drift * self.drift_magnitude * (2 * torch.rand(D, device=proprio.device) - 1)
            if batch:
                drift = drift.view(1, T, D).expand(B, -1, -1)
            else:
                drift = drift.view(T, D)
        
        elif self.drift_type == 'random_walk':
            # Random walk drift
            steps = torch.randn(T, D, device=proprio.device) * self.drift_magnitude * 0.1
            drift = torch.cumsum(steps, dim=0)
            if batch:
                drift = drift.unsqueeze(0).expand(B, -1, -1)
        
        else:
            return proprio
        
        return proprio + drift


class ProprioceptionAugmentation:
    """
    Combined proprioception augmentation.
    
    Includes:
    - Sensor noise injection
    - Joint dropout
    - Temporal jitter
    - Coordinate perturbation
    - Drift simulation
    """
    
    def __init__(
        self,
        config: Optional[ProprioAugConfig] = None,
        p: float = 0.5,
    ):
        """
        Args:
            config: Proprioception augmentation configuration
            p: Probability of applying augmentation
        """
        self.config = config or ProprioAugConfig()
        self.p = p
        
        # Initialize sub-augmentations
        self.sensor_noise = SensorNoiseAugmentation(
            noise_std=self.config.noise_std,
            noise_type=self.config.noise_type,
        )
        self.joint_dropout = JointDropout(
            dropout_p=self.config.joint_dropout_p,
            min_joints_kept=self.config.min_joints_kept,
        )
        self.temporal_jitter = TemporalJitter(
            jitter_ms=self.config.temporal_jitter_ms,
            sample_rate=self.config.sample_rate,
        )
        self.coord_perturb = CoordinatePerturbation(
            position_noise=self.config.position_noise,
            velocity_noise=self.config.velocity_noise,
            acceleration_noise=self.config.acceleration_noise,
            orientation_noise=self.config.orientation_noise,
        )
        self.drift_sim = DriftSimulation(
            drift_prob=self.config.drift_prob,
            drift_magnitude=self.config.drift_magnitude,
        )
    
    def __call__(
        self,
        proprio: torch.Tensor,
        use_noise: bool = True,
        use_dropout: bool = True,
        use_jitter: bool = True,
        use_perturbation: bool = True,
        use_drift: bool = True,
    ) -> torch.Tensor:
        """
        Apply proprioception augmentation.
        
        Args:
            proprio: [T, D] or [B, T, D] proprioception data
            use_noise: Apply sensor noise
            use_dropout: Apply joint dropout
            use_jitter: Apply temporal jitter
            use_perturbation: Apply coordinate perturbation
            use_drift: Apply drift simulation
            
        Returns:
            Augmented proprioception data
        """
        if random.random() > self.p:
            return proprio
        
        # Apply scaling
        if random.random() < 0.3:
            scale = random.uniform(*self.config.scale_range)
            proprio = proprio * scale
        
        # Sensor noise
        if use_noise and random.random() < 0.7:
            proprio = self.sensor_noise(proprio)
        
        # Joint dropout
        if use_dropout and random.random() < 0.3:
            proprio = self.joint_dropout(proprio)
        
        # Temporal jitter
        if use_jitter and random.random() < 0.3:
            proprio = self.temporal_jitter(proprio)
        
        # Coordinate perturbation
        if use_perturbation and random.random() < 0.5:
            proprio = self.coord_perturb(proprio)
        
        # Drift simulation
        if use_drift:
            proprio = self.drift_sim(proprio)
        
        return proprio
