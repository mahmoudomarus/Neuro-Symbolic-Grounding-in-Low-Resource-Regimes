"""
SOTA Encoder: Pre-trained Vision Transformer (DINO ViT) for zero-shot visual understanding.

This replaces the custom ResNet encoder with a state-of-the-art pre-trained model that
already understands geometry, depth, and semantic concepts without any training.

Physical intuition: This is the "adult eyes" - a model that has seen millions of images
and learned rich visual representations. The [CLS] token acts as a universal concept vector.

Run from project root: python -c "from src.world_model.encoder_sota import SOTA_Encoder; print(SOTA_Encoder())"
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class SOTA_Encoder(nn.Module):
    """
    Pre-trained Vision Transformer encoder using DINO (self-distillation).
    
    DINO (Emerging Properties in Self-Supervised Vision Transformers) learns rich
    visual representations that capture semantic similarity without labels. The [CLS]
    token provides a global image representation suitable for zero-shot recognition.
    
    Physical intuition: Unlike our custom encoder that learns "concept islands" from
    scratch, this model arrives with innate knowledge - it already knows what makes
    a dog look like a dog, even if it's never been explicitly labeled.
    
    Args:
        model_name: Timm model identifier (default: 'vit_base_patch16_224.dino')
        freeze: If True, freeze all weights (no training, pure inference)
        embed_dim: Output embedding dimension (default: 768 for ViT-Base)
    """
    
    def __init__(
        self,
        model_name: str = "vit_base_patch16_224.dino",
        freeze: bool = True,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        
        try:
            import timm
        except ImportError:
            raise ImportError(
                "timm is required for SOTA_Encoder. Install with: pip install timm"
            )
        
        # Load pre-trained DINO ViT model
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.embed_dim = embed_dim
        self.model_name = model_name
        
        # Freeze weights if specified (for zero-shot inference)
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()
        
        # Get the actual output dimension from the model
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            dummy_output = self.model(dummy_input)
            actual_dim = dummy_output.shape[1]
        
        # If output dim doesn't match expected, add projection layer
        if actual_dim != embed_dim:
            self.projection = nn.Linear(actual_dim, embed_dim)
        else:
            self.projection = nn.Identity()
    
    @property
    def out_channels(self) -> int:
        """Return output embedding dimension (for compatibility with old encoder API)."""
        return self.embed_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode image to global concept vector.
        
        Args:
            x: Input images [B, 3, H, W] (any size, will be resized to 224x224)
        
        Returns:
            Global concept vector [B, embed_dim] representing the image semantics
        """
        # Resize to 224x224 if needed (DINO ViT expects this size)
        if x.shape[2] != 224 or x.shape[3] != 224:
            x = torch.nn.functional.interpolate(
                x, size=(224, 224), mode='bilinear', align_corners=False
            )
        
        # Forward through ViT (returns [CLS] token by default with num_classes=0)
        features = self.model(x)  # [B, actual_dim]
        
        # Project to target dimension if needed
        features = self.projection(features)  # [B, embed_dim]
        
        return features
    
    def get_patch_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get spatial patch embeddings (for visualization or spatial reasoning).
        
        Args:
            x: Input images [B, 3, H, W]
        
        Returns:
            Patch embeddings [B, num_patches, embed_dim]
        """
        # Resize to 224x224 if needed
        if x.shape[2] != 224 or x.shape[3] != 224:
            x = torch.nn.functional.interpolate(
                x, size=(224, 224), mode='bilinear', align_corners=False
            )
        
        # Forward through patch embedding and transformer blocks
        # This gets all patch tokens (not just [CLS])
        x = self.model.patch_embed(x)
        
        # Add position embeddings
        if hasattr(self.model, 'pos_embed'):
            x = x + self.model.pos_embed
        
        # Forward through transformer blocks
        for blk in self.model.blocks:
            x = blk(x)
        
        x = self.model.norm(x)
        
        # Return all patch tokens (excluding [CLS] which is at index 0)
        return x[:, 1:, :]  # [B, num_patches, embed_dim]


class SOTA_EncoderWithProjection(nn.Module):
    """
    DINO ViT encoder with custom projection head for compatibility.
    
    Use this when you need to match a specific output dimension (e.g., 64 for
    compatibility with existing ConceptBinder trained on custom data).
    
    Args:
        base_model_name: Timm model identifier
        target_dim: Target output dimension (e.g., 64)
        freeze_base: If True, freeze base model weights
    """
    
    def __init__(
        self,
        base_model_name: str = "vit_base_patch16_224.dino",
        target_dim: int = 64,
        freeze_base: bool = True,
    ) -> None:
        super().__init__()
        
        # Load base DINO encoder (768-D output)
        self.base_encoder = SOTA_Encoder(
            model_name=base_model_name,
            freeze=freeze_base,
            embed_dim=768,
        )
        
        # Add projection head to target dimension
        self.projection = nn.Sequential(
            nn.Linear(768, 384),
            nn.ReLU(inplace=True),
            nn.Linear(384, target_dim),
        )
        
        self.target_dim = target_dim
    
    @property
    def out_channels(self) -> int:
        """Return output embedding dimension."""
        return self.target_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode image to target dimension concept vector.
        
        Args:
            x: Input images [B, 3, H, W]
        
        Returns:
            Concept vector [B, target_dim]
        """
        features = self.base_encoder(x)  # [B, 768]
        projected = self.projection(features)  # [B, target_dim]
        return projected


def get_sota_encoder(
    freeze: bool = True,
    target_dim: Optional[int] = None,
) -> nn.Module:
    """
    Factory function to create the appropriate SOTA encoder.
    
    Args:
        freeze: If True, freeze base model weights
        target_dim: If specified, add projection to this dimension; else return 768-D
    
    Returns:
        SOTA encoder module
    """
    if target_dim is None or target_dim == 768:
        return SOTA_Encoder(freeze=freeze, embed_dim=768)
    else:
        return SOTA_EncoderWithProjection(
            target_dim=target_dim,
            freeze_base=freeze,
        )


if __name__ == "__main__":
    # Quick test
    print("Testing SOTA_Encoder...")
    encoder = SOTA_Encoder()
    print(f"Model: {encoder.model_name}")
    print(f"Output dim: {encoder.out_channels}")
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 224, 224)
    output = encoder(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test with projection
    print("\nTesting SOTA_EncoderWithProjection...")
    encoder_proj = SOTA_EncoderWithProjection(target_dim=64)
    output_proj = encoder_proj(dummy_input)
    print(f"Projected output shape: {output_proj.shape}")
    
    print("\nAll tests passed!")
