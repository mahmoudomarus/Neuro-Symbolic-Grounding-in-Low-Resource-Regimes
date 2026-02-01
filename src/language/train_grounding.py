"""
Language grounding: align word vectors (ConceptBinder) with visual vectors (encoder).

Trains only the ConceptBinder so that binder(label) is close in cosine similarity
to pooled encoder(images of that label). Encoder is frozen (Phase 5 weights).

Logic switch:
- If ./data/my_dataset exists: uses custom RGB data, dynamic num_classes from dataset.classes
- Else: uses Fashion-MNIST demo mode (10 classes)

Saves checkpoints/dataset_config.json with input_channels, num_classes, class_names
so dashboard and main can load the correct configuration.

Run from project root: python src/language/train_grounding.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision

from src.language.binder import ConceptBinder
from src.world_model.config import EncoderConfig, RealWorldConfig
from src.world_model.encoder import SpatialEncoder
from src.data.custom_loader import CustomImageDataset, custom_dataset_exists

# Constants
CUSTOM_DATA_ROOT = "./data/my_dataset"
FASHION_LABELS = [
    "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def get_demo_encoder_config() -> EncoderConfig:
    """Config for Fashion-MNIST demo mode (1 channel, 28x28)."""
    return EncoderConfig(
        input_channels=1,
        base_channels=32,
        num_blocks=2,
        output_channels=64,
        strides_per_stage=(2, 2),
        use_geometry=False,
    )


def get_custom_encoder_config() -> EncoderConfig:
    """Config for custom RGB data (3 channels, 64x64)."""
    real_world = RealWorldConfig()
    return EncoderConfig(
        input_channels=real_world.input_channels,  # 3 for RGB
        base_channels=32,
        num_blocks=2,
        output_channels=64,
        strides_per_stage=(2, 2),
        use_geometry=False,
    )


def save_dataset_config(
    input_channels: int,
    num_classes: int,
    class_names: List[str],
    checkpoint_dir: Path,
) -> None:
    """
    Save dataset configuration to JSON for dashboard/main to load.
    
    Args:
        input_channels: Number of input channels (1 for grayscale, 3 for RGB)
        num_classes: Number of classes in the dataset
        class_names: List of class name strings
        checkpoint_dir: Directory to save config file
    """
    config = {
        "input_channels": input_channels,
        "num_classes": num_classes,
        "class_names": class_names,
    }
    config_path = checkpoint_dir / "dataset_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved dataset config to {config_path}")


def train_grounding_demo(
    encoder: SpatialEncoder,
    binder: ConceptBinder,
    data_path: str = "./data",
    batch_size: int = 128,
    epochs: int = 5,
    lr: float = 1e-3,
    device: torch.device | None = None,
) -> None:
    """Train binder on Fashion-MNIST (demo mode)."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = encoder.to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    binder = binder.to(device)
    optimizer = torch.optim.Adam(binder.parameters(), lr=lr)

    train_set = torchvision.datasets.FashionMNIST(
        root=data_path,
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)

    for epoch in range(epochs):
        binder.train()
        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                v = encoder(images)
            v = v.mean(dim=(2, 3))
            v = F.normalize(v, dim=1)

            t = binder(labels)

            sim = F.cosine_similarity(v, t, dim=1)
            loss = (1 - sim).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (batch_idx + 1) % 100 == 0:
                print(f"  Epoch {epoch + 1} batch {batch_idx + 1} | loss = {loss.item():.4f}")

    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(binder.state_dict(), ckpt_dir / "binder_v1.pth")
    print(f"Saved binder to checkpoints/binder_v1.pth")
    
    # Save dataset config for demo mode
    save_dataset_config(
        input_channels=1,
        num_classes=10,
        class_names=FASHION_LABELS,
        checkpoint_dir=ckpt_dir,
    )


def train_grounding_custom(
    encoder: SpatialEncoder,
    binder: ConceptBinder,
    dataset: CustomImageDataset,
    batch_size: int = 128,
    epochs: int = 5,
    lr: float = 1e-3,
    device: torch.device | None = None,
) -> None:
    """Train binder on custom RGB dataset."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = encoder.to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    binder = binder.to(device)
    optimizer = torch.optim.Adam(binder.parameters(), lr=lr)
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    print(f"Training binder on custom dataset: {len(dataset)} images, {len(dataset.classes)} classes")
    print(f"Classes: {dataset.classes}")

    for epoch in range(epochs):
        binder.train()
        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                v = encoder(images)
            v = v.mean(dim=(2, 3))
            v = F.normalize(v, dim=1)

            t = binder(labels)

            sim = F.cosine_similarity(v, t, dim=1)
            loss = (1 - sim).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (batch_idx + 1) % 100 == 0:
                print(f"  Epoch {epoch + 1} batch {batch_idx + 1} | loss = {loss.item():.4f}")

    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(binder.state_dict(), ckpt_dir / "binder_v1.pth")
    print(f"Saved binder to checkpoints/binder_v1.pth")
    
    # Save dataset config for custom mode
    save_dataset_config(
        input_channels=3,
        num_classes=len(dataset.classes),
        class_names=list(dataset.classes),
        checkpoint_dir=ckpt_dir,
    )


def main() -> None:
    checkpoint_path = Path("checkpoints/encoder_v1.pth")
    if not checkpoint_path.exists():
        print("Encoder checkpoint not found. Run: python src/world_model/train_encoder.py")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedding_dim = 64  # Must match encoder output_channels
    
    # Logic switch: Custom RGB data vs Fashion-MNIST demo
    if custom_dataset_exists(CUSTOM_DATA_ROOT):
        print("=" * 60)
        print("CUSTOM DATA MODE: Found ./data/my_dataset")
        print("Training binder on RGB images (3 channels, 64x64)")
        print("=" * 60)
        
        enc_config = get_custom_encoder_config()
        encoder = SpatialEncoder(enc_config, geometry_config=None)
        encoder.load_state_dict(torch.load(checkpoint_path, map_location=device))
        
        # Load custom dataset to get class count
        real_world = RealWorldConfig()
        dataset = CustomImageDataset(CUSTOM_DATA_ROOT, resolution=real_world.input_resolution)
        num_classes = len(dataset.classes)
        
        binder = ConceptBinder(num_classes=num_classes, embedding_dim=embedding_dim)
        
        train_grounding_custom(
            encoder,
            binder,
            dataset=dataset,
            batch_size=128,
            epochs=5,
            lr=1e-3,
            device=device,
        )
    else:
        print("=" * 60)
        print("RUNNING IN DEMO MODE: No custom dataset found")
        print("Training binder on Fashion-MNIST (1 channel, 28x28)")
        print("To use custom data, create ./data/my_dataset with class subfolders")
        print("=" * 60)
        
        enc_config = get_demo_encoder_config()
        encoder = SpatialEncoder(enc_config, geometry_config=None)
        encoder.load_state_dict(torch.load(checkpoint_path, map_location=device))
        
        num_classes = 10  # Fashion-MNIST
        binder = ConceptBinder(num_classes=num_classes, embedding_dim=embedding_dim)
        
        train_grounding_demo(
            encoder,
            binder,
            data_path="./data",
            batch_size=128,
            epochs=5,
            lr=1e-3,
            device=device,
        )


if __name__ == "__main__":
    main()
