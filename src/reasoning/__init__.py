"""
Causal Reasoning Layer (Layer 2).

Understands WHY things happen, not just WHAT happens next:
- Cause → Effect relationships
- Intuitive physics (gravity, collision, support)
- Counterfactual reasoning ("what if...")
- Intervention vs observation distinction

Key insight: Causality is learned through intervention (doing),
not just observation (seeing).

ARCHITECTURAL UPDATE (Peer Review):
Physics priors are now ADAPTIVE, not hard-coded. They use:
- Learnable prior weights (start at 0.9, floor at 0.3)
- Residual correction networks for exceptions
- Critical period regularization to preserve core physics
"""

from .causal_layer import (
    CausalReasoner,
    CausalConfig,
    CausalRelation,
)
from .intuitive_physics import (
    IntuitivePhysics,
    PhysicsViolation,
    PhysicsLaw,
    AdaptivePhysicsPrior,
    AdaptiveSolidityPrior,
    GravityPrior,
    SolidityPrior,
    ContactCausalityPrior,
)
from .counterfactual import CounterfactualReasoner

__all__ = [
    "CausalReasoner",
    "CausalConfig",
    "CausalRelation",
    "IntuitivePhysics",
    "PhysicsViolation",
    "PhysicsLaw",
    "AdaptivePhysicsPrior",
    "AdaptiveSolidityPrior",
    "GravityPrior",
    "SolidityPrior",
    "ContactCausalityPrior",
    "CounterfactualReasoner",
]
