"""
Sensory system: maps raw observations (e.g. pixels) to a spatial latent map.

SpatialEncoder uses ResNet-style CNN blocks to produce a SpatialMap [B, C, H, W],
preserving spatial relationships. It then applies RotaryEmbedding2D so the model
has innate spatial awareness—same concept at different positions is the same
concept, just translated.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .config import EncoderConfig, GeometryConfig
from .geometry import GeometricBias


class ResBlock(nn.Module):
    """
    Residual block (Conv-BN-ReLU-Conv-BN + skip). Preserves spatial resolution
    when stride=1; used as the main building block for the sensory stack.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return self.relu(out)


class SpatialEncoder(nn.Module):
    """
    Encodes visual (or sensory) patches into a latent spatial map.

    Uses a CNN backbone (ResNet-style) so the output is a SpatialMap [B, C, H, W],
    not a flat vector—preserving where things are. Then applies a geometric prior
    (RotaryEmbedding2D) so position is encoded as a fundamental truth: the same
    object at (0,0) vs (1,1) is the same concept, just moved.
    """

    def __init__(
        self,
        encoder_config: EncoderConfig,
        geometry_config: Optional[GeometryConfig] = None,
    ) -> None:
        super().__init__()
        self.encoder_config = encoder_config
        self.geometry_config = geometry_config

        ic = encoder_config.input_channels
        bc = encoder_config.base_channels
        oc = encoder_config.output_channels
        strides = encoder_config.strides_per_stage
        n_blocks = encoder_config.num_blocks

        # Stem: first conv + bn + relu; first element of strides applies here
        s0 = strides[0] if strides else 1
        self.stem = nn.Sequential(
            nn.Conv2d(ic, bc, kernel_size=7, stride=s0, padding=3),
            nn.BatchNorm2d(bc),
            nn.ReLU(inplace=True),
        )

        # Stages: each stage has n_blocks ResBlocks; first block of stage can downsample
        stages_list: list[nn.Module] = []
        in_ch = bc
        rest_strides = list(strides[1:]) if len(strides) > 1 else []
        for i, stride in enumerate(rest_strides):
            out_ch = min(bc * (2 ** (i + 1)), oc)
            blocks: list[nn.Module] = []
            for j in range(n_blocks):
                blocks.append(ResBlock(in_ch, out_ch, stride=(stride if j == 0 else 1)))
                in_ch = out_ch  # next block in same stage gets this stage's out_ch
            stages_list.append(nn.Sequential(*blocks))
        if not stages_list:
            out_ch = min(bc * 2, oc)
            stages_list = [nn.Sequential(*[ResBlock(bc, out_ch, stride=1) for _ in range(n_blocks)])]
            in_ch = out_ch
        self.stages = nn.Sequential(*stages_list)

        # Final projection to output_channels if needed
        if in_ch != oc:
            self.proj = nn.Conv2d(in_ch, oc, kernel_size=1)
        else:
            self.proj = nn.Identity()

        self._out_channels = oc
        self._geometry: Optional[GeometricBias] = None
        if encoder_config.use_geometry and geometry_config is not None:
            if geometry_config.dim != oc:
                raise ValueError(
                    f"GeometryConfig.dim ({geometry_config.dim}) must equal "
                    f"EncoderConfig.output_channels ({oc})."
                )
            self._geometry = GeometricBias(geometry_config)

    @property
    def out_channels(self) -> int:
        return self._out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Map observation to spatial latent map.

        Args:
            x: Observation tensor (B, input_channels, H, W), e.g. pixels.

        Returns:
            Spatial map (B, output_channels, H', W') with spatial structure
            and geometric prior applied.
        """
        x = self.stem(x)
        x = self.stages(x)
        z = self.proj(x)
        if self._geometry is not None:
            z = self._geometry(z)
        return z

    def output_shape(self, input_height: int, input_width: int) -> Tuple[int, int, int]:
        """
        Return (C, H_out, W_out) for the given input spatial size (for downstream
        modules that need to know the latent map size).
        """
        with torch.no_grad():
            dummy = torch.zeros(1, self.encoder_config.input_channels, input_height, input_width)
            out = self.forward(dummy)
            _, c, h, w = out.shape
        return (c, h, w)
