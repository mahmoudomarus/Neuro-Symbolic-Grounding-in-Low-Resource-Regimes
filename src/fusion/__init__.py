"""
Cross-modal fusion module.

Provides mechanisms to fuse information from multiple modalities
(vision, audio, proprioception) into unified representations.
"""

from .cross_modal import (
    CrossModalFusion,
    CrossModalAttentionLayer,
    ModalityEmbedding,
)

__all__ = [
    "CrossModalFusion",
    "CrossModalAttentionLayer",
    "ModalityEmbedding",
]
