"""
Proprioceptive encoder for body state.

Encodes body position, velocity, acceleration, and orientation
into a latent representation that can be fused with vision and audio.

Proprioception is the "sixth sense" - the sense of body position and movement.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ProprioEncoderConfig:
    """Configuration for proprioceptive encoder."""
    # Input dimensions (default: position + velocity + acceleration + orientation)
    # position: [x, y, z] = 3
    # velocity: [vx, vy, vz] = 3
    # acceleration: [ax, ay, az] = 3
    # orientation: [roll, pitch, yaw] = 3
    # Total: 12 dimensions
    input_dim: int = 12
    hidden_dim: int = 128
    output_dim: int = 256
    num_layers: int = 3
    use_layer_norm: bool = True
    dropout: float = 0.1


class ProprioEncoder(nn.Module):
    """
    Encodes body state (position, velocity, acceleration, orientation).
    
    Input format: [x, y, z, vx, vy, vz, ax, ay, az, roll, pitch, yaw] = 12 dims
    
    The encoder handles both single timesteps and sequences.
    """
    
    def __init__(self, config: ProprioEncoderConfig) -> None:
        super().__init__()
        
        self.config = config
        
        # Build MLP layers
        layers = []
        in_dim = config.input_dim
        
        for i in range(config.num_layers - 1):
            out_dim = config.hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            
            if config.use_layer_norm:
                layers.append(nn.LayerNorm(out_dim))
            
            layers.append(nn.GELU())
            
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
            
            in_dim = out_dim
        
        # Final projection
        layers.append(nn.Linear(in_dim, config.output_dim))
        
        self.encoder = nn.Sequential(*layers)
        
        self.out_channels = config.output_dim
    
    def forward(self, body_state: torch.Tensor) -> torch.Tensor:
        """
        Encode body state to latent representation.
        
        Args:
            body_state: Body state tensor
                - [B, input_dim] for single timestep
                - [B, T, input_dim] for sequence
                
        Returns:
            Encoded features with same batch/time structure but output_dim channels
        """
        if body_state.dim() == 2:
            # Single timestep: [B, input_dim] -> [B, output_dim]
            return self.encoder(body_state)
        
        elif body_state.dim() == 3:
            # Sequence: [B, T, input_dim] -> [B, T, output_dim]
            B, T, D = body_state.shape
            
            # Flatten batch and time
            x = body_state.reshape(B * T, D)
            
            # Encode
            x = self.encoder(x)
            
            # Reshape back
            return x.reshape(B, T, -1)
        
        else:
            raise ValueError(f"Expected 2D or 3D input, got {body_state.dim()}D")
    
    @staticmethod
    def create_body_state(
        position: torch.Tensor,
        velocity: Optional[torch.Tensor] = None,
        acceleration: Optional[torch.Tensor] = None,
        orientation: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Create body state tensor from components.
        
        Args:
            position: Position [B, 3] or [B, T, 3]
            velocity: Velocity [B, 3] or [B, T, 3] (optional)
            acceleration: Acceleration [B, 3] or [B, T, 3] (optional)
            orientation: Orientation [B, 3] or [B, T, 3] (optional)
            
        Returns:
            Body state tensor [B, 12] or [B, T, 12]
        """
        components = [position]
        
        if velocity is not None:
            components.append(velocity)
        else:
            components.append(torch.zeros_like(position))
        
        if acceleration is not None:
            components.append(acceleration)
        else:
            components.append(torch.zeros_like(position))
        
        if orientation is not None:
            components.append(orientation)
        else:
            components.append(torch.zeros_like(position))
        
        return torch.cat(components, dim=-1)


class ProprioEncoderWithMemory(nn.Module):
    """
    Proprioceptive encoder with temporal memory (RNN).
    
    This version maintains a hidden state to track body movement
    over time, useful for understanding motion trajectories.
    """
    
    def __init__(self, config: ProprioEncoderConfig) -> None:
        super().__init__()
        
        self.config = config
        
        # Initial embedding
        self.embed = nn.Linear(config.input_dim, config.hidden_dim)
        
        # GRU for temporal processing
        self.rnn = nn.GRU(
            input_size=config.hidden_dim,
            hidden_size=config.hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=config.dropout if config.dropout > 0 else 0,
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.output_dim),
        )
        
        self.out_channels = config.output_dim
    
    def forward(
        self, 
        body_state: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode body state sequence with memory.
        
        Args:
            body_state: Body state [B, T, input_dim]
            hidden: Previous hidden state [2, B, hidden_dim] (optional)
            
        Returns:
            Tuple of:
            - Encoded features [B, T, output_dim]
            - Hidden state [2, B, hidden_dim]
        """
        # Embed
        x = self.embed(body_state)  # [B, T, hidden_dim]
        
        # RNN
        x, hidden = self.rnn(x, hidden)  # [B, T, hidden_dim]
        
        # Project
        output = self.output_proj(x)  # [B, T, output_dim]
        
        return output, hidden
    
    def get_initial_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Get initial hidden state."""
        return torch.zeros(
            2, batch_size, self.config.hidden_dim,
            device=device
        )


class IMUEncoder(nn.Module):
    """
    Encoder specifically for IMU (Inertial Measurement Unit) data.
    
    IMU data typically includes:
    - 3-axis accelerometer
    - 3-axis gyroscope
    - Sometimes 3-axis magnetometer
    """
    
    def __init__(
        self,
        input_dim: int = 6,  # 3 accel + 3 gyro
        output_dim: int = 128,
        window_size: int = 50,  # ~0.5 second at 100Hz
    ) -> None:
        super().__init__()
        
        self.window_size = window_size
        
        # 1D conv over time
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.GELU(),
            
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            
            nn.Conv1d(64, output_dim, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(output_dim),
            nn.GELU(),
        )
        
        self.out_channels = output_dim
    
    def forward(self, imu_data: torch.Tensor) -> torch.Tensor:
        """
        Encode IMU data.
        
        Args:
            imu_data: IMU readings [B, T, input_dim]
            
        Returns:
            Encoded features [B, output_dim]
        """
        # Reshape for conv1d: [B, input_dim, T]
        x = imu_data.permute(0, 2, 1)
        
        # Encode
        x = self.encoder(x)  # [B, output_dim, T']
        
        # Global average pool
        return x.mean(dim=2)  # [B, output_dim]
