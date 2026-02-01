"""
Concept Binder: maps class labels to the same vector space as the SpatialEncoder.

Learns the 'ideal Platonic form' of each object class so the agent can 'imagine'
an object given its name. Output dimension must match the encoder's pooled output.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConceptBinder(nn.Module):
    """
    Maps class indices to normalized vectors in the encoder's latent space.

    Uses a simple embedding table so that label k (e.g. 7 = Sneaker) maps to a
    vector that can be aligned with pooled visual features via cosine similarity.
    Output dim must match encoder output_channels (pooled).
    """

    def __init__(self, num_classes: int, embedding_dim: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_classes, embedding_dim)

    def forward(self, class_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            class_indices: Long tensor of shape (B,) or (B, 1) with values in [0, num_classes-1].

        Returns:
            Normalized vectors of shape (B, embedding_dim).
        """
        if class_indices.dim() > 1:
            class_indices = class_indices.squeeze(-1)
        out = self.embedding(class_indices)
        return F.normalize(out, dim=1)
