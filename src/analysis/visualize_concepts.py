"""
Visualize the encoder's concept space: PCA to 2D and scatter by Fashion-MNIST label.

Proves the model grouped concepts (e.g. Shoes vs Shirts) without being told labels.
Run from project root: python src/analysis/visualize_concepts.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from src.world_model.config import EncoderConfig
from src.world_model.encoder import SpatialEncoder

# Fashion-MNIST class names for legend
FASHION_LABELS = [
    "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def get_encoder_config() -> EncoderConfig:
    """Same config as train_encoder.py (Fashion-MNIST, 1 channel, 64 output)."""
    return EncoderConfig(
        input_channels=1,
        base_channels=32,
        num_blocks=2,
        output_channels=64,
        strides_per_stage=(2, 2),
        use_geometry=False,
    )


def visualize_latent_space(checkpoint_path: str | Path, output_path: str | Path = "concept_map.png") -> None:
    """
    Load encoder, run 1000 test images through it, pool to [1000, C], PCA to 2D, scatter by label.
    Saves plot to concept_map.png.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}. Run train_encoder.py first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc_config = get_encoder_config()
    encoder = SpatialEncoder(enc_config, geometry_config=None)
    encoder.load_state_dict(torch.load(checkpoint_path, map_location=device))
    encoder.eval()

    # Load Fashion-MNIST test set (1000 images)
    test_set = torchvision.datasets.FashionMNIST(
        root="./data",
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )
    # Get 1000 samples (random from test set)
    n_samples = 1000
    indices = torch.randperm(len(test_set))[:n_samples].tolist()
    images = torch.stack([test_set[i][0] for i in indices])
    labels = [test_set[i][1] for i in indices]

    with torch.no_grad():
        z = encoder(images.to(device))
    # Pool: [1000, C, H, W] -> [1000, C]
    z_pooled = z.mean(dim=(2, 3)).cpu().numpy()
    labels_np = [labels[i] for i in range(n_samples)]

    # PCA to 2D
    pca = PCA(n_components=2)
    z_2d = pca.fit_transform(z_pooled)

    # Scatter: color by label (0-9)
    fig, ax = plt.subplots(figsize=(10, 8))
    for label_id in range(10):
        mask = [l == label_id for l in labels_np]
        ax.scatter(
            z_2d[mask, 0],
            z_2d[mask, 1],
            label=FASHION_LABELS[label_id],
            alpha=0.6,
            s=20,
        )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Concept Space (Encoder Latent → PCA 2D)")
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved concept map to {output_path}")


if __name__ == "__main__":
    visualize_latent_space("checkpoints/encoder_v1.pth")
