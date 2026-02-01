"""
Entry point for the Neuro-Symbolic JEPA agent.

Uses trained encoder + ConceptBinder + EpisodicMemory. Run 'Life Log': watch 20 images, then recall by label.
Run from project root: python src/main.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on path when running python src/main.py
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import torch
import torch.nn.functional as F
from torchvision import transforms
import torchvision

from src.language.binder import ConceptBinder
from src.memory.episodic import EpisodicMemory
from src.world_model.config import EncoderConfig
from src.world_model.encoder import SpatialEncoder

# Fashion-MNIST class names
FASHION_LABELS = [
    "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def get_trained_encoder_config() -> EncoderConfig:
    """Same config as train_encoder.py (Fashion-MNIST, 1 channel, 64 output)."""
    return EncoderConfig(
        input_channels=1,
        base_channels=32,
        num_blocks=2,
        output_channels=64,
        strides_per_stage=(2, 2),
        use_geometry=False,
    )


def run_life_log(
    encoder: torch.nn.Module,
    binder: ConceptBinder,
    memory: EpisodicMemory,
    device: torch.device,
) -> None:
    """Life Log: watch 20 images (movie), store in memory, then recall by label."""
    test_set = torchvision.datasets.FashionMNIST(
        root="./data",
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )

    memory.clear()
    num_steps = 20

    # The Movie: 20 random images from test set
    indices = torch.randperm(len(test_set))[:num_steps].tolist()

    for step in range(num_steps):
        img, _ = test_set[indices[step]]
        obs = img.unsqueeze(0).to(device)

        with torch.no_grad():
            z = encoder(obs)
        z_pooled = z.mean(dim=(2, 3))
        z_pooled = F.normalize(z_pooled, dim=1)

        memory.store(z_pooled, step, metadata=None)
        print(f"[Step {step}] Observing...")

    # The Interrogation
    print()
    print("What do you want to find in my memory? (e.g., Sneaker)")
    print("0=T-shirt, 1=Trouser, 2=Pullover, 3=Dress, 4=Coat,")
    print("5=Sandal, 6=Shirt, 7=Sneaker, 8=Bag, 9=Boot)")
    try:
        raw = input("Label (0-9): ").strip()
        target_label = int(raw)
    except (ValueError, EOFError):
        target_label = 7
        print("Using default: 7 (Sneaker)")
    if target_label < 0 or target_label > 9:
        target_label = 7
        print("Using default: 7 (Sneaker)")

    target_name = FASHION_LABELS[target_label]
    query = binder(torch.tensor([target_label], device=device))
    matches = memory.recall(query, threshold=0.8)

    if matches:
        steps_str = " and ".join(f"Step {s}" for s, _ in matches)
        print(f"I remember seeing a {target_name} at {steps_str}.")
    else:
        print("I don't recall seeing that.")


def main() -> None:
    encoder_path = Path("checkpoints/encoder_v1.pth")
    binder_path = Path("checkpoints/binder_v1.pth")
    if not encoder_path.exists():
        print("Encoder not found. Run: python src/world_model/train_encoder.py")
        sys.exit(1)
    if not binder_path.exists():
        print("Binder not found. Run: python src/language/train_grounding.py")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc_config = get_trained_encoder_config()
    encoder = SpatialEncoder(enc_config, geometry_config=None)
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    encoder = encoder.to(device)
    encoder.eval()

    binder = ConceptBinder(num_classes=10, embedding_dim=64)
    binder.load_state_dict(torch.load(binder_path, map_location=device))
    binder = binder.to(device)
    binder.eval()

    memory = EpisodicMemory()

    print("Neuro-Symbolic JEPA Agent — Life Log (Episodic Memory)")
    print("=" * 60)
    run_life_log(encoder, binder, memory, device)
    print("=" * 60)


if __name__ == "__main__":
    main()
