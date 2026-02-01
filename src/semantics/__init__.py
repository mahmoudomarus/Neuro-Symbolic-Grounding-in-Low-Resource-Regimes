"""
Semantic Properties Layer (Layer 1).

Extracts meaningful properties from perceptual world state:
- Physical properties: hardness, weight, size, temperature
- Categorical properties: animate/inanimate, agent/object
- Affordances: what can be done with objects

Key insight: Properties are PERCEIVED from multi-modal input,
not learned from language labels.

ARCHITECTURAL UPDATE (Peer Review):
1. DynamicPropertyBank for open-ended property discovery
2. VisualDynamicsPropertyLearner for learning weight from video
3. Physics grounding to avoid learning from static images alone
"""

from .property_layer import (
    PropertyLayer,
    PropertyVector,
    PropertyConfig,
    DynamicPropertyBank,
    SlotAttention,
    RobustSlotAttention,
)
from .affordances import AffordanceDetector
from .categories import CategoryClassifier
from .physics_grounding import (
    VisualDynamicsPropertyLearner,
    PhysicsGroundedPropertyLayer,
    PhysicsGroundingConfig,
    OpticalFlowTracker,
)

__all__ = [
    "PropertyLayer",
    "PropertyVector",
    "PropertyConfig",
    "DynamicPropertyBank",
    "SlotAttention",
    "RobustSlotAttention",
    "AffordanceDetector",
    "CategoryClassifier",
    "VisualDynamicsPropertyLearner",
    "PhysicsGroundedPropertyLayer",
    "PhysicsGroundingConfig",
    "OpticalFlowTracker",
]
