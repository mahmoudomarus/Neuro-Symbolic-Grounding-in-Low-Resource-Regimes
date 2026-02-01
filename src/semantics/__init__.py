"""
Semantic Properties Layer (Layer 1).

Extracts meaningful properties from perceptual world state:
- Physical properties: hardness, weight, size, temperature
- Categorical properties: animate/inanimate, agent/object
- Affordances: what can be done with objects

Key insight: Properties are PERCEIVED from multi-modal input,
not learned from language labels.
"""

from .property_layer import (
    PropertyLayer,
    PropertyVector,
    PropertyConfig,
)
from .affordances import AffordanceDetector
from .categories import CategoryClassifier

__all__ = [
    "PropertyLayer",
    "PropertyVector",
    "PropertyConfig",
    "AffordanceDetector",
    "CategoryClassifier",
]
