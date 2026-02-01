"""
Kindergarten training: SimCLR contrastive learning on Fashion-MNIST or custom RGB data.

Trains the SpatialEncoder via a SimCLR wrapper (projection head + NT-Xent loss).
Saves only the encoder weights (no projection head) to checkpoints.

Logic switch:
- If ./data/my_dataset exists with valid class subfolders: uses RGB pipeline (3 channels, 64x64)
- Else: uses Fashion-MNIST demo mode (1 channel, 28x28)

Run from project root: python src/world_model/train_encoder.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms

from src.world_model.config import EncoderConfig, TrainingConfig, RealWorldConfig
from src.world_model.encoder import SpatialEncoder
from src.data.custom_loader import SimCLRCustomDataset, custom_dataset_exists

# Constants
CUSTOM_DATA_ROOT = "./data/my_dataset"


# --- Step A: SimCLR Wrapper ---


class ProjectionHead(nn.Module):
    """MLP: Linear -> ReLU -> Linear for SimCLR projection."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: Optional[int] = None) -> None:
        super().__init__()
        hidden_dim = hidden_dim or in_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class SimCLRWrapper(nn.Module):
    """
    Wraps SpatialEncoder with global average pooling and a projection head.
    Forward: encoder(x) -> [B, C, H, W] -> pool -> [B, C] -> projection -> z [B, projection_dim].
    """

    def __init__(self, encoder: SpatialEncoder, projection_dim: int) -> None:
        super().__init__()
        self.encoder = encoder
        encoder_dim = encoder.out_channels
        self.projection_head = ProjectionHead(encoder_dim, projection_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder: [B, C_in, H, W] -> [B, C, H', W']
        h = self.encoder(x)
        # Global average pooling: [B, C, H', W'] -> [B, C]
        h = h.mean(dim=(2, 3))
        # Projection: [B, C] -> [B, projection_dim]
        z = self.projection_head(h)
        return z


# --- Step B: NT-Xent contrastive loss ---


def nt_xent_loss(z_i: torch.Tensor, z_j: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    NT-Xent (normalized temperature-scaled cross entropy) loss for SimCLR.
    z_i, z_j: [N, D] two augmented views of the same batch. Positive pair: (z_i[k], z_j[k]).
    """
    N = z_i.shape[0]
    z = torch.cat([z_i, z_j], dim=0)  # [2N, D]
    z = F.normalize(z, dim=1)
    sim = torch.mm(z, z.t()) / temperature  # [2N, 2N]
    # Mask out self (diagonal)
    mask = torch.eye(2 * N, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, float("-inf"))
    # Positive indices: for i in [0, N), pos[i] = i+N; for i in [N, 2N), pos[i] = i-N
    pos_idx = torch.cat([torch.arange(N, 2 * N, device=z.device), torch.arange(N, device=z.device)])
    # For each row i, take exp(sim[i, pos_idx[i]]) as numerator
    logits = sim
    labels = pos_idx
    # Cross entropy: -log(exp(logits[i, labels[i]]) / sum_j exp(logits[i,j]))
    # For row i, we want -log(exp(sim[i, pos_idx[i]]) / sum_j exp(sim[i,j]))
    # log_softmax then nll_loss, or manually
    log_probs = F.log_softmax(logits, dim=1)
    loss = -log_probs[torch.arange(2 * N, device=z.device), labels].mean()
    return loss


# --- Step C: Data loading and augmentation ---


class SimCLRTransform:
    """
    For every image, returns two augmented views (x_i, x_j).
    Augmentations: RandomResizedCrop(28), RandomHorizontalFlip, RandomRotation(10).
    """

    def __init__(self, size: int = 28) -> None:
        self.size = size
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])

    def __call__(self, img: object) -> tuple[torch.Tensor, torch.Tensor]:
        x_i = self.transform(img)
        x_j = self.transform(img)
        return x_i, x_j


class SimCLRFashionMNIST(Dataset):
    """Fashion-MNIST with SimCLR transform returning (x_i, x_j) per sample."""

    def __init__(self, root: str, transform: SimCLRTransform | None = None) -> None:
        self.fmnist = torchvision.datasets.FashionMNIST(
            root=root,
            train=True,
            download=True,
            transform=None,
        )
        self.transform = transform or SimCLRTransform(28)

    def __len__(self) -> int:
        return len(self.fmnist)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img, _ = self.fmnist[idx]
        return self.transform(img)


# --- Step D: Training loop ---


def train_simclr(
    encoder: SpatialEncoder,
    train_config: TrainingConfig,
    encoder_config: EncoderConfig,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimCLRWrapper(encoder, train_config.projection_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.learning_rate)

    dataset = SimCLRFashionMNIST(train_config.dataset_path, transform=SimCLRTransform(28))
    loader = DataLoader(
        dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    global_step = 0
    for epoch in range(train_config.epochs):
        model.train()
        for batch_idx, (x_i, x_j) in enumerate(loader):
            x_i, x_j = x_i.to(device), x_j.to(device)
            z_i = model(x_i)
            z_j = model(x_j)
            loss = nt_xent_loss(z_i, z_j, train_config.temperature)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1
            if global_step % 100 == 0:
                print(f"  Step {global_step} (epoch {epoch + 1}) | loss = {loss.item():.4f}")

    # Save only encoder weights (discard projection head)
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "encoder_v1.pth"
    torch.save(encoder.state_dict(), ckpt_path)
    print(f"Saved encoder weights to {ckpt_path}")


def get_custom_encoder_config() -> EncoderConfig:
    """
    Build EncoderConfig from RealWorldConfig for custom RGB data.
    
    Returns EncoderConfig suitable for 3-channel, 64x64 RGB images.
    """
    real_world = RealWorldConfig()
    return EncoderConfig(
        input_channels=real_world.input_channels,  # 3 for RGB
        base_channels=32,
        num_blocks=2,
        output_channels=64,
        strides_per_stage=(2, 2),
        use_geometry=False,
    )


def get_demo_encoder_config() -> EncoderConfig:
    """
    Build EncoderConfig for Fashion-MNIST demo mode.
    
    Returns EncoderConfig suitable for 1-channel, 28x28 grayscale images.
    """
    return EncoderConfig(
        input_channels=1,
        base_channels=32,
        num_blocks=2,
        output_channels=64,
        strides_per_stage=(2, 2),
        use_geometry=False,
    )


def train_simclr_custom(
    encoder: SpatialEncoder,
    train_config: TrainingConfig,
) -> None:
    """
    Train encoder on custom RGB dataset using SimCLR.
    
    Uses SimCLRCustomDataset which provides two augmented views per image.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimCLRWrapper(encoder, train_config.projection_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.learning_rate)
    
    real_world = RealWorldConfig()
    dataset = SimCLRCustomDataset(CUSTOM_DATA_ROOT, resolution=real_world.input_resolution)
    loader = DataLoader(
        dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    
    print(f"Training on custom dataset: {len(dataset)} images, {len(dataset.classes)} classes")
    print(f"Classes: {dataset.classes}")
    
    global_step = 0
    for epoch in range(train_config.epochs):
        model.train()
        for batch_idx, (x_i, x_j) in enumerate(loader):
            x_i, x_j = x_i.to(device), x_j.to(device)
            z_i = model(x_i)
            z_j = model(x_j)
            loss = nt_xent_loss(z_i, z_j, train_config.temperature)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1
            if global_step % 100 == 0:
                print(f"  Step {global_step} (epoch {epoch + 1}) | loss = {loss.item():.4f}")
    
    # Save only encoder weights (discard projection head)
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "encoder_v1.pth"
    torch.save(encoder.state_dict(), ckpt_path)
    print(f"Saved encoder weights to {ckpt_path}")


def main() -> None:
    train_config = TrainingConfig()
    
    # Logic switch: Custom RGB data vs Fashion-MNIST demo
    if custom_dataset_exists(CUSTOM_DATA_ROOT):
        print("=" * 60)
        print("CUSTOM DATA MODE: Found ./data/my_dataset")
        print("Training encoder on RGB images (3 channels, 64x64)")
        print("=" * 60)
        
        encoder_config = get_custom_encoder_config()
        encoder = SpatialEncoder(encoder_config, geometry_config=None)
        train_simclr_custom(encoder, train_config)
    else:
        print("=" * 60)
        print("RUNNING IN DEMO MODE: No custom dataset found")
        print("Training encoder on Fashion-MNIST (1 channel, 28x28)")
        print("To use custom data, create ./data/my_dataset with class subfolders")
        print("=" * 60)
        
        encoder_config = get_demo_encoder_config()
        encoder = SpatialEncoder(encoder_config, geometry_config=None)
        train_simclr(encoder, train_config, encoder_config)


if __name__ == "__main__":
    main()
