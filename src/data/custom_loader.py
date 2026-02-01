"""
Custom data loader for RGB images from user-provided datasets.

Provides:
- CustomImageDataset: Standard ImageFolder loader with strict transforms
- SimCLRCustomDataset: Two-view dataset for contrastive learning (SimCLR)

Expected directory structure:
    ./data/my_dataset/
        class_a/
            img1.jpg
            img2.png
            ...
        class_b/
            img1.jpg
            ...

Run from project root: python -c "from src.data import CustomImageDataset; print(len(CustomImageDataset()))"
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple, Optional, Callable, List

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image

# ImageNet normalization statistics (standard for RGB pretrained models)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_strict_transform(resolution: int = 64) -> transforms.Compose:
    """
    Standard inference/evaluation transform for custom RGB images.
    
    Applies: Resize -> CenterCrop -> ToTensor -> Normalize (ImageNet stats)
    
    Args:
        resolution: Target spatial size (default 64).
    
    Returns:
        Composed transform pipeline.
    """
    return transforms.Compose([
        transforms.Resize(resolution),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_simclr_transform(resolution: int = 64) -> transforms.Compose:
    """
    SimCLR augmentation transform for contrastive learning on RGB images.
    
    Applies: RandomResizedCrop -> RandomHorizontalFlip -> ColorJitter -> ToTensor -> Normalize
    
    Args:
        resolution: Target spatial size (default 64).
    
    Returns:
        Composed transform pipeline (apply twice per image for two views).
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(resolution, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def validate_dataset_root(root: Path) -> List[str]:
    """
    Validate that a dataset root directory exists and has proper structure.
    
    Args:
        root: Path to dataset root (e.g., ./data/my_dataset)
    
    Returns:
        List of class names (subfolder names)
    
    Raises:
        FileNotFoundError: If root doesn't exist, has no subfolders, or subfolders have no images
    """
    if not root.exists():
        raise FileNotFoundError(
            f"Dataset root not found: {root}\n"
            f"Create {root} with subfolders per class (e.g., dog/, cat/, car/).\n"
            f"Each subfolder should contain images (.jpg, .png, .jpeg, .gif, .bmp, .webp)."
        )
    
    # Find subdirectories (potential class folders)
    subdirs = [d for d in root.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    if not subdirs:
        raise FileNotFoundError(
            f"No class subfolders found in: {root}\n"
            f"Add at least one class subfolder with images.\n"
            f"Example structure:\n"
            f"  {root}/\n"
            f"    class_a/\n"
            f"      image1.jpg\n"
            f"    class_b/\n"
            f"      image1.jpg"
        )
    
    # Check each subfolder has at least one image
    valid_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
    classes_with_images = []
    
    for subdir in subdirs:
        has_images = any(
            f.suffix.lower() in valid_extensions
            for f in subdir.iterdir()
            if f.is_file()
        )
        if has_images:
            classes_with_images.append(subdir.name)
    
    if not classes_with_images:
        raise FileNotFoundError(
            f"No images found in any class subfolder of: {root}\n"
            f"Add .jpg, .png, or other image files to class folders.\n"
            f"Found folders: {[d.name for d in subdirs]}"
        )
    
    return sorted(classes_with_images)


class CustomImageDataset(ImageFolder):
    """
    Custom RGB image dataset based on ImageFolder structure.
    
    Loads images from ./data/my_dataset (or specified root) where each subfolder
    is a class. Applies strict transforms: Resize -> CenterCrop -> ToTensor -> Normalize.
    
    Physical intuition: This is the agent's 'eye' for real-world RGB data. Each image
    is normalized to ImageNet statistics so the encoder sees consistent value ranges.
    
    Args:
        root: Path to dataset root (default: ./data/my_dataset)
        resolution: Target spatial size (default: 64)
        transform: Optional custom transform (overrides default strict transform)
    
    Raises:
        FileNotFoundError: If root doesn't exist or has no valid class subfolders
    """
    
    def __init__(
        self,
        root: str = "./data/my_dataset",
        resolution: int = 64,
        transform: Optional[Callable] = None,
    ) -> None:
        root_path = Path(root)
        
        # Validate dataset structure before initializing ImageFolder
        validate_dataset_root(root_path)
        
        # Use default strict transform if none provided
        if transform is None:
            transform = get_strict_transform(resolution)
        
        super().__init__(root=str(root_path), transform=transform)
        
        self.resolution = resolution
        self._class_names = self.classes  # ImageFolder populates self.classes


class SimCLRCustomDataset(Dataset):
    """
    Two-view dataset for SimCLR contrastive learning on custom RGB images.
    
    For each image, returns two different augmented views (x_i, x_j) for
    contrastive learning. Uses strong augmentations (crop, flip, color jitter).
    
    Physical intuition: This teaches the encoder that two different views of
    the same object are 'the same concept', while views of different objects
    are 'different concepts'.
    
    Args:
        root: Path to dataset root (default: ./data/my_dataset)
        resolution: Target spatial size (default: 64)
    
    Raises:
        FileNotFoundError: If root doesn't exist or has no valid class subfolders
    """
    
    def __init__(
        self,
        root: str = "./data/my_dataset",
        resolution: int = 64,
    ) -> None:
        root_path = Path(root)
        
        # Validate dataset structure
        self._class_names = validate_dataset_root(root_path)
        
        # Load ImageFolder without transform (we apply transforms manually for two views)
        self._dataset = ImageFolder(root=str(root_path), transform=None)
        self._transform = get_simclr_transform(resolution)
        self.resolution = resolution
        self.classes = self._dataset.classes
    
    def __len__(self) -> int:
        return len(self._dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns two augmented views of the same image.
        
        Args:
            idx: Index into dataset
        
        Returns:
            (x_i, x_j): Two different augmented views of the same image,
                        each of shape [C, H, W] normalized to ImageNet stats.
        """
        img, _ = self._dataset[idx]  # PIL Image, label
        
        # Ensure RGB (convert grayscale if needed)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Apply transform twice to get two different views
        x_i = self._transform(img)
        x_j = self._transform(img)
        
        return x_i, x_j


def custom_dataset_exists(root: str = "./data/my_dataset") -> bool:
    """
    Check if a valid custom dataset exists at the specified path.
    
    Args:
        root: Path to check
    
    Returns:
        True if root exists and has at least one class subfolder with images
    """
    try:
        validate_dataset_root(Path(root))
        return True
    except FileNotFoundError:
        return False


if __name__ == "__main__":
    # Quick validation script
    root = "./data/my_dataset"
    if custom_dataset_exists(root):
        dataset = CustomImageDataset(root)
        print(f"Custom dataset found: {len(dataset)} images, {len(dataset.classes)} classes")
        print(f"Classes: {dataset.classes}")
        
        # Test one sample
        img, label = dataset[0]
        print(f"Sample shape: {img.shape}, label: {label}")
    else:
        print(f"No custom dataset at {root}")
        print("Create the directory structure as described in the docstring.")
