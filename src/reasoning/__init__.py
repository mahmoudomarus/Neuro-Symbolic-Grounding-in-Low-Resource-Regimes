"""
Causal Reasoning Layer (Layer 2).

Understands WHY things happen, not just WHAT happens next:
- Cause → Effect relationships
- Intuitive physics (gravity, collision, support)
- Counterfactual reasoning ("what if...")
- Intervention vs observation distinction

Key insight: Causality is learned through intervention (doing),
not just observation (seeing).
"""

from .causal_layer import (
    CausalReasoner,
    CausalConfig,
    CausalRelation,
)
from .intuitive_physics import (
    IntuitivePhysics,
    PhysicsViolation,
)
from .counterfactual import CounterfactualReasoner

__all__ = [
    "CausalReasoner",
    "CausalConfig",
    "CausalRelation",
    "IntuitivePhysics",
    "PhysicsViolation",
    "CounterfactualReasoner",
]
