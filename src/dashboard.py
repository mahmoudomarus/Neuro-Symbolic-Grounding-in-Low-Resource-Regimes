"""
Phase 9/11: Neuro-Link Dashboard — real-time visualization of the agent's inputs, memory, and decisions.

Supports both:
- Demo mode: Fashion-MNIST (1 channel, 10 classes)
- Custom mode: RGB images from ./data/my_dataset (3 channels, dynamic classes)

Mode is determined by checkpoints/dataset_config.json (written by train_grounding.py).

Run from project root: .venv/bin/streamlit run src/dashboard.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from io import BytesIO

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import streamlit as st
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torchvision import transforms
import torchvision
from PIL import Image

from src.language.binder import ConceptBinder
from src.manager.agent import CognitiveAgent
from src.manager.config import AgentConfig
from src.memory.episodic import EpisodicMemory
from src.tools.library import CalculatorTool, WikiTool
from src.world_model.config import EncoderConfig, RealWorldConfig
from src.world_model.encoder import SpatialEncoder
from src.data.custom_loader import CustomImageDataset, custom_dataset_exists

# Default class labels for demo mode
FASHION_LABELS = [
    "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

# ImageNet normalization stats for RGB
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_dataset_config() -> Dict:
    """
    Load dataset configuration from checkpoints/dataset_config.json.
    
    Returns dict with:
        input_channels: int (1 for grayscale, 3 for RGB)
        num_classes: int
        class_names: List[str]
    
    Falls back to Fashion-MNIST defaults if config not found.
    """
    config_path = Path(_project_root) / "checkpoints" / "dataset_config.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            return json.load(f)
    # Default to Fashion-MNIST demo mode
    return {
        "input_channels": 1,
        "num_classes": 10,
        "class_names": FASHION_LABELS,
    }


def get_encoder_config(dataset_config: Dict) -> EncoderConfig:
    """Build EncoderConfig from dataset configuration."""
    return EncoderConfig(
        input_channels=dataset_config["input_channels"],
        base_channels=32,
        num_blocks=2,
        output_channels=64,
        strides_per_stage=(2, 2),
        use_geometry=False,
    )


class WorldModel:
    def __init__(self, encoder: torch.nn.Module) -> None:
        self.encoder = encoder


@st.cache_resource
def load_encoder(_dataset_config: Dict):
    """Load trained encoder (cached) based on dataset config."""
    path = Path(_project_root) / "checkpoints" / "encoder_v1.pth"
    if not path.exists():
        return None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc_config = get_encoder_config(_dataset_config)
    encoder = SpatialEncoder(enc_config, geometry_config=None)
    encoder.load_state_dict(torch.load(path, map_location=device))
    encoder = encoder.to(device)
    encoder.eval()
    return encoder, device


@st.cache_resource
def load_binder(_dataset_config: Dict):
    """Load trained binder (cached) based on dataset config."""
    path = Path(_project_root) / "checkpoints" / "binder_v1.pth"
    if not path.exists():
        return None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = _dataset_config["num_classes"]
    binder = ConceptBinder(num_classes=num_classes, embedding_dim=64)
    binder.load_state_dict(torch.load(path, map_location=device))
    binder = binder.to(device)
    binder.eval()
    return binder, device


def get_upload_transform(is_rgb: bool, resolution: int = 64) -> transforms.Compose:
    """
    Get transform for uploaded images.
    
    For RGB: uses ImageNet normalization
    For grayscale: uses Fashion-MNIST normalization
    """
    if is_rgb:
        return transforms.Compose([
            transforms.Resize(resolution),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(resolution),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])


def unnormalize_for_display(img_tensor: torch.Tensor, is_rgb: bool) -> np.ndarray:
    """
    Un-normalize a tensor for display (reverse normalization).
    
    Args:
        img_tensor: Tensor of shape [C, H, W]
        is_rgb: If True, use ImageNet stats; else use Fashion-MNIST stats
    
    Returns:
        NumPy array of shape [H, W, C] or [H, W] suitable for display
    """
    img = img_tensor.clone().cpu()
    
    if is_rgb:
        # Un-normalize with ImageNet stats
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        img = img * std + mean
    else:
        # Un-normalize with Fashion-MNIST stats
        img = img * 0.5 + 0.5
    
    # Clamp to [0, 1]
    img = torch.clamp(img, 0, 1)
    
    # Convert to numpy for display
    arr = img.permute(1, 2, 0).numpy()
    if arr.shape[2] == 1:
        arr = arr.squeeze(-1)
    
    return arr


def predict_class(
    encoder: SpatialEncoder,
    binder: ConceptBinder,
    img_tensor: torch.Tensor,
    class_names: List[str],
    device: torch.device,
) -> Tuple[str, float, List[Tuple[str, float]]]:
    """
    Predict class for an image using encoder + binder.
    
    Returns:
        (predicted_class_name, confidence, all_similarities)
    """
    with torch.no_grad():
        z = encoder(img_tensor.unsqueeze(0).to(device))
        z_pooled = F.normalize(z.mean(dim=(2, 3)), dim=1)
        
        similarities = []
        for c in range(len(class_names)):
            q = binder(torch.tensor([c], device=device))
            sim = F.cosine_similarity(z_pooled, q, dim=1).item()
            similarities.append((class_names[c], sim))
        
        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        best_name, best_sim = similarities[0]
        # Convert similarity to confidence percentage
        confidence = max(0, min(100, (best_sim + 1) / 2 * 100))  # Map [-1, 1] to [0, 100]
        
        return best_name, confidence, similarities


def estimated_concept(
    binder: ConceptBinder,
    vector: torch.Tensor,
    device: torch.device,
    class_names: List[str],
) -> str:
    """Find closest class name for a stored vector using the binder."""
    vector = vector.unsqueeze(0)
    best_sim = -2.0
    best_idx = 0
    for c in range(len(class_names)):
        q = binder(torch.tensor([c], device=device)).cpu()
        sim = F.cosine_similarity(vector, q, dim=1).item()
        if sim > best_sim:
            best_sim = sim
            best_idx = c
    return class_names[best_idx]


def parse_query(text: str, class_names: List[str]) -> int | None:
    """Parse user input to label index. Accepts digit or class name."""
    text = text.strip()
    if not text:
        return None
    if text.isdigit():
        i = int(text)
        if 0 <= i < len(class_names):
            return i
        return None
    text_lower = text.lower()
    for i, name in enumerate(class_names):
        if name.lower() == text_lower:
            return i
    return None


def main() -> None:
    st.set_page_config(page_title="Neuro-Link Dashboard", layout="wide")
    st.title("Neuro-Symbolic Agent — Neuro-Link Dashboard")
    
    # Load dataset configuration
    dataset_config = load_dataset_config()
    class_names = dataset_config["class_names"]
    is_rgb = dataset_config["input_channels"] == 3
    is_custom_mode = is_rgb  # Custom mode uses RGB
    
    # Sidebar: Dataset info and Live Test zone
    with st.sidebar:
        st.header("Dataset Configuration")
        
        if is_custom_mode:
            st.success("**Mode: Custom Data**")
            st.write(f"Input: {dataset_config['input_channels']} channels (RGB)")
        else:
            st.info("**Mode: Demo (Fashion-MNIST)**")
            st.write(f"Input: {dataset_config['input_channels']} channel (Grayscale)")
        
        st.write(f"**Classes ({dataset_config['num_classes']}):**")
        for i, name in enumerate(class_names):
            st.caption(f"  {i}: {name}")
        
        st.divider()
        
        # Live Test zone: File uploader
        st.header("Live Test Zone")
        st.caption("Upload an image to classify it")
        
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=["jpg", "jpeg", "png", "gif", "webp", "bmp"],
        )

    # Load models
    encoder_data = load_encoder(dataset_config)
    binder_data = load_binder(dataset_config)
    if encoder_data is None:
        st.error("Encoder not found. Run: python src/world_model/train_encoder.py")
        st.stop()
    if binder_data is None:
        st.error("Binder not found. Run: python src/language/train_grounding.py")
        st.stop()

    encoder, device = encoder_data
    binder, _ = binder_data
    world_model = WorldModel(encoder)
    tools = [WikiTool(64), CalculatorTool(64)]
    agent = CognitiveAgent(world_model, tools, AgentConfig(uncertainty_threshold=0.7))

    # Initialize session state
    if "memory" not in st.session_state:
        st.session_state.memory = EpisodicMemory()
    if "step" not in st.session_state:
        st.session_state.step = 0
    if "memory_display" not in st.session_state:
        st.session_state.memory_display = []
    if "test_set" not in st.session_state:
        # Load appropriate test set based on mode
        if is_custom_mode:
            custom_root = Path(_project_root) / "data" / "my_dataset"
            if custom_root.exists():
                real_world = RealWorldConfig()
                st.session_state.test_set = CustomImageDataset(
                    str(custom_root),
                    resolution=real_world.input_resolution,
                )
            else:
                st.warning("Custom dataset not found. Using Fashion-MNIST for demo.")
                st.session_state.test_set = torchvision.datasets.FashionMNIST(
                    root=str(_project_root / "data"),
                    train=False,
                    download=True,
                    transform=transforms.ToTensor(),
                )
        else:
            st.session_state.test_set = torchvision.datasets.FashionMNIST(
                root=str(_project_root / "data"),
                train=False,
                download=True,
                transform=transforms.ToTensor(),
            )
    if "current_image" not in st.session_state:
        st.session_state.current_image = None
    if "current_uncertainty" not in st.session_state:
        st.session_state.current_uncertainty = 0.0
    if "current_decision" not in st.session_state:
        st.session_state.current_decision = "—"
    if "is_rgb" not in st.session_state:
        st.session_state.is_rgb = is_rgb

    # Process uploaded file in sidebar
    with st.sidebar:
        if uploaded_file is not None:
            # Load and process the uploaded image
            img_pil = Image.open(uploaded_file)
            
            # Convert to RGB if needed (for consistency)
            if is_rgb:
                if img_pil.mode != 'RGB':
                    img_pil = img_pil.convert('RGB')
                resolution = RealWorldConfig().input_resolution
            else:
                # For grayscale mode, convert to L
                if img_pil.mode != 'L':
                    img_pil = img_pil.convert('L')
                resolution = 28
            
            # Apply transform
            transform = get_upload_transform(is_rgb, resolution)
            img_tensor = transform(img_pil)
            
            # Predict class
            pred_class, confidence, all_sims = predict_class(
                encoder, binder, img_tensor, class_names, device
            )
            
            # Display results
            st.subheader("Prediction Result")
            
            # Show uploaded image (original)
            st.image(img_pil, caption="Uploaded Image", use_container_width=True)
            
            # Show prediction
            st.metric("Predicted Class", pred_class)
            st.metric("Confidence", f"{confidence:.1f}%")
            
            # Show top-3 similarities
            st.caption("Top predictions:")
            for name, sim in all_sims[:3]:
                bar_val = max(0, min(1, (sim + 1) / 2))  # Map [-1, 1] to [0, 1]
                st.progress(bar_val, text=f"{name}: {sim:.3f}")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("The World")
        if st.button("Step (Generate Random Event)"):
            test_set = st.session_state.test_set
            idx = torch.randint(0, len(test_set), (1,)).item()
            img, _ = test_set[idx]
            obs = img.unsqueeze(0).to(device)
            with torch.no_grad():
                decision, uncertainty = agent.step(obs)
                z = encoder(obs)
            z_pooled = z.mean(dim=(2, 3))
            z_pooled = F.normalize(z_pooled, dim=1)
            step = st.session_state.step
            st.session_state.memory.store(z_pooled, step, {"uncertainty": uncertainty})
            concept = estimated_concept(binder, z_pooled.squeeze(0).detach().cpu(), device, class_names)
            st.session_state.memory_display.append({
                "Step": step,
                "Estimated Concept": concept,
                "Uncertainty": round(uncertainty, 3),
            })
            st.session_state.step += 1
            st.session_state.current_image = img
            st.session_state.current_uncertainty = uncertainty
            st.session_state.current_decision = decision
            st.rerun()

        if st.session_state.current_image is not None:
            img = st.session_state.current_image
            # Handle both RGB and grayscale display
            arr = unnormalize_for_display(img, st.session_state.is_rgb)
            st.image(arr, caption="Current observation", use_container_width=True)
        else:
            st.caption("Click Step to show an observation.")

        u = st.session_state.current_uncertainty
        st.caption("Uncertainty (Green = Low, Red = High)")
        st.progress(u, text=f"{u:.2f}")
        if u < 0.5:
            st.success("System 1: ACTION (confident)")
        else:
            st.error("System 2: TOOL (uncertain)")
        st.write("Decision:", st.session_state.current_decision)

    with col2:
        st.subheader("The Chat")
        # Dynamic hint based on class names
        hint = f"e.g., {class_names[0]} or 0" if class_names else "e.g., 0"
        query_text = st.text_input(f"Ask me to find something ({hint})")
        if query_text:
            label = parse_query(query_text, class_names)
            if label is not None:
                query_vec = binder(torch.tensor([label], device=device))
                matches = st.session_state.memory.recall(query_vec, threshold=0.8)
                name = class_names[label]
                if matches:
                    steps_str = ", ".join(f"Step {s}" for s, _ in matches)
                    st.success(f"I remember seeing a {name} at {steps_str}.")
                else:
                    st.info("I don't recall seeing that.")
            else:
                st.warning(f"Enter a label 0–{len(class_names)-1} or a class name.")

    with col3:
        st.subheader("The Memory Palace")
        display_list = st.session_state.memory_display[-10:]
        if display_list:
            df = pd.DataFrame(display_list)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.caption("Last 10 memories will appear here after you Step.")
        if st.button("Clear memory"):
            st.session_state.memory.clear()
            st.session_state.memory_display = []
            st.session_state.step = 0
            st.session_state.current_image = None
            st.session_state.current_uncertainty = 0.0
            st.session_state.current_decision = "—"
            st.rerun()


if __name__ == "__main__":
    main()
