"""
SOTA Binder: CLIP for universal language-vision grounding.

This replaces the custom ConceptBinder with OpenAI's CLIP, which was trained on
400 million (image, text) pairs and can understand any concept without fine-tuning.

Physical intuition: This is the "universal translator" - it maps both images and
text to the same vector space, enabling zero-shot recognition of any concept you
can describe in words.

Run from project root: python -c "from src.language.binder_sota import SOTA_Binder; print(SOTA_Binder())"
"""
from __future__ import annotations

from typing import List, Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SOTA_Binder(nn.Module):
    """
    CLIP-based universal language-vision binder.
    
    Unlike ConceptBinder which learns to map 10 class indices to vectors, SOTA_Binder
    can map ANY text description to the same space as images. This enables zero-shot
    recognition: describe what you want to find, and it will find it.
    
    Physical intuition: CLIP learned that "a photo of a cat" and an actual cat image
    are the same concept, just expressed in different modalities. This binder leverages
    that innate knowledge.
    
    Args:
        model_name: CLIP model variant (default: 'ViT-B/32')
        device: Device to run on (cpu/cuda)
        freeze: If True, freeze all weights (no training, pure inference)
    """
    
    def __init__(
        self,
        model_name: str = "ViT-B/32",
        device: Optional[str] = None,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        
        try:
            import open_clip
        except ImportError:
            raise ImportError(
                "open_clip is required for SOTA_Binder. Install with: pip install open-clip-torch"
            )
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        
        # Load CLIP model
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained='openai'
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        
        self.model = self.model.to(self.device)
        
        # Get embedding dimension
        self.embed_dim = self.model.text_projection.shape[1]
        
        # Freeze weights if specified
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()
    
    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension (for compatibility with ConceptBinder API)."""
        return self.embed_dim
    
    def embed_text(self, text: Union[str, List[str]]) -> torch.Tensor:
        """
        Encode text to normalized concept vector(s).
        
        Args:
            text: Single string or list of strings (e.g., ["a red car", "a blue sky"])
        
        Returns:
            Normalized text embeddings [B, embed_dim] where B = len(text)
        """
        if isinstance(text, str):
            text = [text]
        
        # Tokenize text
        tokens = self.tokenizer(text).to(self.device)
        
        # Encode text
        with torch.no_grad():
            text_features = self.model.encode_text(tokens)
            text_features = F.normalize(text_features, dim=-1)
        
        return text_features
    
    def embed_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images to normalized concept vector(s).
        
        Args:
            images: Images [B, 3, H, W] (will be resized to CLIP's expected size)
        
        Returns:
            Normalized image embeddings [B, embed_dim]
        """
        # Resize to CLIP's expected input size (224x224 for most models)
        if images.shape[2] != 224 or images.shape[3] != 224:
            images = F.interpolate(
                images, size=(224, 224), mode='bilinear', align_corners=False
            )
        
        # Encode image
        with torch.no_grad():
            image_features = self.model.encode_image(images)
            image_features = F.normalize(image_features, dim=-1)
        
        return image_features
    
    def forward(self, class_indices: torch.Tensor) -> torch.Tensor:
        """
        For compatibility with ConceptBinder API (not used in SOTA mode).
        
        In SOTA mode, use embed_text() directly with class names instead.
        This method is kept for backward compatibility.
        
        Args:
            class_indices: Ignored in SOTA mode
        
        Returns:
            Dummy tensor (use embed_text() instead)
        """
        raise NotImplementedError(
            "SOTA_Binder does not use class indices. Use embed_text(class_name) instead."
        )
    
    def compute_similarity(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        temperature: float = 0.01,
    ) -> torch.Tensor:
        """
        Compute similarity between image and text features.
        
        Args:
            image_features: Normalized image embeddings [B, embed_dim]
            text_features: Normalized text embeddings [N, embed_dim]
            temperature: Temperature scaling (lower = sharper probabilities)
        
        Returns:
            Similarity logits [B, N]
        """
        # Compute cosine similarity (dot product of normalized vectors)
        logits = (image_features @ text_features.T) / temperature
        return logits
    
    def classify(
        self,
        images: torch.Tensor,
        class_names: List[str],
        templates: Optional[List[str]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Zero-shot classification using text prompts.
        
        Args:
            images: Images [B, 3, H, W]
            class_names: List of class names (e.g., ["cat", "dog", "car"])
            templates: Optional prompt templates (e.g., ["a photo of a {}"])
        
        Returns:
            (predicted_class_indices, probabilities) where:
                predicted_class_indices: [B] indices into class_names
                probabilities: [B, N] softmax probabilities over N classes
        """
        # Default template
        if templates is None:
            templates = ["a photo of a {}"]
        
        # Encode images
        image_features = self.embed_image(images)  # [B, embed_dim]
        
        # Encode all class names with all templates
        text_features_list = []
        for class_name in class_names:
            # Average over all templates for this class
            class_texts = [template.format(class_name) for template in templates]
            class_features = self.embed_text(class_texts)  # [num_templates, embed_dim]
            class_features = class_features.mean(dim=0, keepdim=True)  # [1, embed_dim]
            text_features_list.append(class_features)
        
        text_features = torch.cat(text_features_list, dim=0)  # [N, embed_dim]
        text_features = F.normalize(text_features, dim=-1)
        
        # Compute similarity and convert to probabilities
        logits = self.compute_similarity(image_features, text_features)
        probs = F.softmax(logits, dim=-1)
        predicted_indices = torch.argmax(probs, dim=-1)
        
        return predicted_indices, probs


def get_sota_binder(
    model_name: str = "ViT-B/32",
    device: Optional[str] = None,
    freeze: bool = True,
) -> SOTA_Binder:
    """
    Factory function to create SOTA binder.
    
    Args:
        model_name: CLIP model variant
        device: Device to run on
        freeze: If True, freeze all weights
    
    Returns:
        SOTA_Binder instance
    """
    return SOTA_Binder(model_name=model_name, device=device, freeze=freeze)


if __name__ == "__main__":
    # Quick test
    print("Testing SOTA_Binder...")
    binder = SOTA_Binder()
    print(f"Model: {binder.model_name}")
    print(f"Embedding dim: {binder.embedding_dim}")
    print(f"Device: {binder.device}")
    
    # Test text embedding
    text = ["a red ferrari", "a cute cat", "a platypus"]
    text_features = binder.embed_text(text)
    print(f"\nText embeddings shape: {text_features.shape}")
    
    # Test image embedding
    dummy_images = torch.randn(2, 3, 224, 224).to(binder.device)
    image_features = binder.embed_image(dummy_images)
    print(f"Image embeddings shape: {image_features.shape}")
    
    # Test classification
    class_names = ["dog", "cat", "car", "plane"]
    pred_indices, probs = binder.classify(dummy_images, class_names)
    print(f"\nClassification results:")
    print(f"Predicted indices: {pred_indices}")
    print(f"Probabilities shape: {probs.shape}")
    print(f"Top prediction for image 0: {class_names[pred_indices[0].item()]} ({probs[0, pred_indices[0]].item():.2%})")
    
    print("\nAll tests passed!")
