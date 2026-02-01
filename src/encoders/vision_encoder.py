"""
Vision encoder: DINO ViT-B/16 for extracting visual features.

Uses pre-trained DINO (self-distillation with no labels) model:
- Model: facebook/dino-vitb16
- Pre-trained on: ImageNet without labels (self-supervised)
- Output: 768-dimensional features

Why DINO:
- Learns object-centric representations without supervision
- Attention maps naturally segment objects
- Strong zero-shot transfer capability
- Matches the project's "grounded" philosophy

Note: The encoder is frozen (no gradient computation) and used as a fixed feature extractor.
Training focuses on learning cross-modal bindings, not re-learning vision.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig


class DinoVisionEncoder(nn.Module):
    """
    DINO ViT-B/16 vision encoder wrapper.
    
    Loads pre-trained DINO model from HuggingFace and extracts features.
    The model is kept frozen - only used for inference.
    """
    
    def __init__(
        self,
        model_name: str = "facebook/dino-vitb16",
        output_dim: int = 768,
        freeze: bool = True,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize DINO vision encoder.
        
        Args:
            model_name: HuggingFace model name (default: facebook/dino-vitb16)
            output_dim: Output feature dimension (768 for ViT-B/16)
            freeze: Whether to freeze model parameters (default: True)
            device: Device to load model on
        """
        super().__init__()
        
        self.model_name = model_name
        self.output_dim = output_dim
        self.freeze = freeze
        
        # Load pre-trained model
        print(f"Loading DINO vision encoder: {model_name}")
        try:
            self.vit = ViTModel.from_pretrained(model_name)
        except Exception as e:
            print(f"Error loading model, trying fallback: {e}")
            # Fallback: create config and load manually
            config = ViTConfig.from_pretrained(model_name)
            self.vit = ViTModel(config)
        
        # Freeze parameters if specified
        if freeze:
            for param in self.vit.parameters():
                param.requires_grad = False
            self.vit.eval()
            print("Vision encoder frozen for inference only")
        
        self.patch_size = 16
        self.image_size = 224
        
        # Move to device
        if device is not None:
            self.vit = self.vit.to(device)
    
    def forward(
        self,
        images: torch.Tensor,
        return_patches: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Extract features from images.
        
        Args:
            images: [B, 3, H, W] RGB images (expected 224x224)
            return_patches: If True, also return patch features
            
        Returns:
            If return_patches=False: [B, 768] CLS token features
            If return_patches=True: Dict with 'cls' and 'patches' keys
        """
        # Ensure correct size if needed
        B, C, H, W = images.shape
        if H != self.image_size or W != self.image_size:
            images = torch.nn.functional.interpolate(
                images,
                size=(self.image_size, self.image_size),
                mode='bilinear',
                align_corners=False
            )
        
        # Normalize images to [-1, 1] for DINO
        # DINO uses ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
        images = (images - mean) / std
        
        with torch.set_grad_enabled(not self.freeze):
            outputs = self.vit(pixel_values=images)
        
        # CLS token: [B, 768]
        cls_features = outputs.last_hidden_state[:, 0]
        
        if return_patches:
            # Patch tokens: [B, 196, 768] (for 224x224 with 16x16 patches)
            patch_features = outputs.last_hidden_state[:, 1:]
            return {
                'cls': cls_features,
                'patches': patch_features,
            }
        
        return cls_features
    
    def get_attention_maps(
        self,
        images: torch.Tensor,
        layer_idx: int = -1,
    ) -> torch.Tensor:
        """
        Extract attention maps for visualization.
        
        Args:
            images: [B, 3, H, W] RGB images
            layer_idx: Which encoder layer to extract attention from (-1 = last)
            
        Returns:
            Attention maps: [B, num_heads, num_patches+1, num_patches+1]
        """
        if self.vit.config.output_attentions:
            with torch.no_grad():
                outputs = self.vit(
                    pixel_values=images,
                    output_attentions=True,
                )
            
            # outputs.attentions is a tuple of attention maps for each layer
            attention = outputs.attentions[layer_idx]
            return attention
        else:
            # If not configured to output attentions, return None
            # You may need to reload the model with output_attentions=True
            return None
    
    @property
    def num_parameters(self) -> int:
        """Return number of trainable/frozen parameters."""
        return sum(p.numel() for p in self.vit.parameters())


class VisionEncoderWithPrior(nn.Module):
    """
    Vision encoder that applies Gabor prior before DINO encoding.
    
    This mimics the biological processing pipeline:
    1. Retinal/cortical preprocessing (Gabor filters)
    2. Higher-level visual processing (DINO)
    """
    
    def __init__(
        self,
        use_gabor: bool = True,
        gabor_orientations: int = 8,
        gabor_scales: int = 5,
        model_name: str = "facebook/dino-vitb16",
        freeze: bool = True,
    ):
        """
        Initialize vision encoder with optional Gabor prior.
        
        Args:
            use_gabor: Whether to apply Gabor filterbank prior
            gabor_orientations: Number of Gabor orientations
            gabor_scales: Number of Gabor scales
            model_name: DINO model name
            freeze: Whether to freeze DINO
        """
        super().__init__()
        
        self.use_gabor = use_gabor
        
        if use_gabor:
            from ..priors import GaborFilterbank
            self.gabor = GaborFilterbank(
                num_orientations=gabor_orientations,
                num_scales=gabor_scales,
            )
            # Project Gabor output from [B, C*num_filters, H, W] to [B, 3, H, W]
            num_gabor_filters = gabor_orientations * gabor_scales * 2
            self.gabor_proj = nn.Conv2d(
                3 * num_gabor_filters,
                3,
                kernel_size=1,
            )
        
        self.encoder = DinoVisionEncoder(
            model_name=model_name,
            freeze=freeze,
        )
    
    def forward(
        self,
        images: torch.Tensor,
        return_patches: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Extract features with optional Gabor preprocessing.
        
        Args:
            images: [B, 3, H, W] RGB images
            return_patches: Whether to return patch features
            
        Returns:
            Features (and optionally patches)
        """
        if self.use_gabor:
            # Apply Gabor filters
            gabor_response = self.gabor(images)
            # Project back to 3 channels
            images = self.gabor_proj(gabor_response)
        
        return self.encoder(images, return_patches=return_patches)
