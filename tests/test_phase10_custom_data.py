"""
Tests for Phase 10: Custom Data Pipeline.

Validates:
- CustomImageDataset initialization and structure
- SimCLRCustomDataset two-view output
- Error handling for missing/invalid datasets
"""
from __future__ import annotations

import sys
from pathlib import Path
import tempfile
import shutil

import pytest
import torch
from PIL import Image

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.data.custom_loader import (
    CustomImageDataset,
    SimCLRCustomDataset,
    validate_dataset_root,
    custom_dataset_exists,
    get_strict_transform,
    get_simclr_transform,
)


class TestValidateDatasetRoot:
    """Tests for validate_dataset_root function."""
    
    def test_raises_on_missing_root(self, tmp_path: Path) -> None:
        """Raises FileNotFoundError if root doesn't exist."""
        nonexistent = tmp_path / "nonexistent"
        with pytest.raises(FileNotFoundError, match="Dataset root not found"):
            validate_dataset_root(nonexistent)
    
    def test_raises_on_empty_root(self, tmp_path: Path) -> None:
        """Raises FileNotFoundError if root has no subfolders."""
        empty_root = tmp_path / "empty_dataset"
        empty_root.mkdir()
        with pytest.raises(FileNotFoundError, match="No class subfolders found"):
            validate_dataset_root(empty_root)
    
    def test_raises_on_no_images(self, tmp_path: Path) -> None:
        """Raises FileNotFoundError if subfolders have no images."""
        root = tmp_path / "dataset"
        root.mkdir()
        (root / "class_a").mkdir()
        (root / "class_b").mkdir()
        # No images in either folder
        with pytest.raises(FileNotFoundError, match="No images found"):
            validate_dataset_root(root)
    
    def test_returns_class_names(self, tmp_path: Path) -> None:
        """Returns sorted list of class names with images."""
        root = tmp_path / "dataset"
        root.mkdir()
        
        # Create two class folders with images
        class_a = root / "apple"
        class_b = root / "banana"
        class_a.mkdir()
        class_b.mkdir()
        
        # Create dummy images
        img = Image.new('RGB', (64, 64), color='red')
        img.save(class_a / "img1.jpg")
        img.save(class_b / "img1.png")
        
        classes = validate_dataset_root(root)
        assert classes == ["apple", "banana"]


class TestCustomImageDataset:
    """Tests for CustomImageDataset."""
    
    @pytest.fixture
    def valid_dataset_path(self, tmp_path: Path) -> Path:
        """Create a temporary valid dataset structure."""
        root = tmp_path / "my_dataset"
        root.mkdir()
        
        for class_name in ["cat", "dog"]:
            class_dir = root / class_name
            class_dir.mkdir()
            for i in range(3):
                img = Image.new('RGB', (100, 100), color='blue')
                img.save(class_dir / f"img{i}.jpg")
        
        return root
    
    def test_loads_valid_dataset(self, valid_dataset_path: Path) -> None:
        """Successfully loads a valid dataset."""
        dataset = CustomImageDataset(str(valid_dataset_path), resolution=64)
        assert len(dataset) == 6  # 2 classes * 3 images
        assert len(dataset.classes) == 2
        assert "cat" in dataset.classes
        assert "dog" in dataset.classes
    
    def test_output_shape(self, valid_dataset_path: Path) -> None:
        """Output tensors have correct shape [C, H, W]."""
        dataset = CustomImageDataset(str(valid_dataset_path), resolution=64)
        img, label = dataset[0]
        assert img.shape == (3, 64, 64)
        assert isinstance(label, int)
        assert 0 <= label < 2
    
    def test_raises_on_invalid_root(self, tmp_path: Path) -> None:
        """Raises FileNotFoundError for invalid root."""
        with pytest.raises(FileNotFoundError):
            CustomImageDataset(str(tmp_path / "nonexistent"))


class TestSimCLRCustomDataset:
    """Tests for SimCLRCustomDataset."""
    
    @pytest.fixture
    def valid_dataset_path(self, tmp_path: Path) -> Path:
        """Create a temporary valid dataset structure."""
        root = tmp_path / "my_dataset"
        root.mkdir()
        
        class_dir = root / "objects"
        class_dir.mkdir()
        for i in range(5):
            img = Image.new('RGB', (100, 100), color='green')
            img.save(class_dir / f"img{i}.jpg")
        
        return root
    
    def test_returns_two_views(self, valid_dataset_path: Path) -> None:
        """Each sample returns two augmented views."""
        dataset = SimCLRCustomDataset(str(valid_dataset_path), resolution=64)
        x_i, x_j = dataset[0]
        
        assert isinstance(x_i, torch.Tensor)
        assert isinstance(x_j, torch.Tensor)
        assert x_i.shape == (3, 64, 64)
        assert x_j.shape == (3, 64, 64)
    
    def test_views_are_different(self, valid_dataset_path: Path) -> None:
        """Two views of the same image should be different (due to augmentation)."""
        dataset = SimCLRCustomDataset(str(valid_dataset_path), resolution=64)
        x_i, x_j = dataset[0]
        
        # Due to random augmentations, views should differ
        # (extremely unlikely to be identical)
        assert not torch.allclose(x_i, x_j, atol=1e-3)
    
    def test_length_matches_images(self, valid_dataset_path: Path) -> None:
        """Dataset length matches number of images."""
        dataset = SimCLRCustomDataset(str(valid_dataset_path), resolution=64)
        assert len(dataset) == 5


class TestTransforms:
    """Tests for transform functions."""
    
    def test_strict_transform_output_shape(self) -> None:
        """Strict transform produces correct output shape."""
        transform = get_strict_transform(resolution=64)
        img = Image.new('RGB', (100, 80), color='red')
        out = transform(img)
        assert out.shape == (3, 64, 64)
    
    def test_simclr_transform_output_shape(self) -> None:
        """SimCLR transform produces correct output shape."""
        transform = get_simclr_transform(resolution=64)
        img = Image.new('RGB', (100, 80), color='red')
        out = transform(img)
        assert out.shape == (3, 64, 64)
    
    def test_strict_transform_is_deterministic(self) -> None:
        """Strict transform produces same output for same input (no randomness)."""
        transform = get_strict_transform(resolution=64)
        img = Image.new('RGB', (100, 80), color='red')
        out1 = transform(img)
        out2 = transform(img)
        assert torch.allclose(out1, out2)


class TestCustomDatasetExists:
    """Tests for custom_dataset_exists helper."""
    
    def test_returns_false_for_missing(self, tmp_path: Path) -> None:
        """Returns False if path doesn't exist."""
        assert custom_dataset_exists(str(tmp_path / "nonexistent")) is False
    
    def test_returns_true_for_valid(self, tmp_path: Path) -> None:
        """Returns True for valid dataset structure."""
        root = tmp_path / "dataset"
        root.mkdir()
        class_dir = root / "class_a"
        class_dir.mkdir()
        img = Image.new('RGB', (64, 64), color='red')
        img.save(class_dir / "img.jpg")
        
        assert custom_dataset_exists(str(root)) is True
