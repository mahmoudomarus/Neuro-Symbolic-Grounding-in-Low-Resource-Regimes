"""
Intuitive Physics - Innate understanding of physical laws.

Humans have innate intuitions about physics:
- Objects fall (gravity)
- Objects don't pass through each other (solidity)
- Unsupported objects fall (support)
- Objects continue moving (inertia)

Infants show surprise when these are violated, suggesting
these are partially innate, not fully learned.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicsLaw(Enum):
    """Fundamental physics intuitions."""
    GRAVITY = "gravity"           # Unsupported objects fall
    SOLIDITY = "solidity"         # Objects don't pass through each other
    SUPPORT = "support"           # Objects need support to stay up
    INERTIA = "inertia"          # Moving objects keep moving
    CONTACT = "contact"          # Causation requires contact
    CONTAINMENT = "containment"  # Objects stay in containers


@dataclass
class PhysicsViolation:
    """A detected violation of intuitive physics."""
    law: PhysicsLaw
    severity: float  # 0-1, how much it violates
    description: str
    

class GravityPrior(nn.Module):
    """
    Innate understanding of gravity.
    
    Expectations:
    - Unsupported objects accelerate downward
    - Objects on support stay put
    - Heavier objects exert more force
    """
    
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        
        # Detect support
        self.support_detector = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid(),
        )
        
        # Predict expected motion under gravity
        self.motion_predictor = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Linear(feature_dim // 2, 3),  # dx, dy, dz
        )
    
    def forward(
        self,
        object_state: torch.Tensor,
        scene_context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict expected motion under gravity.
        
        Returns:
            - is_supported: [B] boolean
            - expected_motion: [B, 3] expected displacement
        """
        is_supported = self.support_detector(object_state).squeeze(-1)
        expected_motion = self.motion_predictor(object_state)
        
        # If supported, expected motion is zero
        expected_motion = expected_motion * (1 - is_supported.unsqueeze(-1))
        
        # Gravity acts downward (negative y typically)
        gravity_bias = torch.tensor([0.0, -1.0, 0.0], device=object_state.device)
        expected_motion = expected_motion + (1 - is_supported.unsqueeze(-1)) * gravity_bias
        
        return is_supported > 0.5, expected_motion


class SolidityPrior(nn.Module):
    """
    Innate understanding that solid objects can't pass through each other.
    
    Detects when two objects would overlap, which violates solidity.
    """
    
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        
        # Overlap detector
        self.overlap_detector = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, 1),
            nn.Sigmoid(),
        )
    
    def forward(
        self,
        object1_state: torch.Tensor,
        object2_state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Detect if two objects would violate solidity.
        
        Returns:
            Violation probability [B]
        """
        combined = torch.cat([object1_state, object2_state], dim=-1)
        return self.overlap_detector(combined).squeeze(-1)


class ContactCausalityPrior(nn.Module):
    """
    Innate understanding that causation requires contact.
    
    "Spooky action at a distance" violates physics intuition.
    If object A moves and object B was nowhere near it, A didn't cause B's motion.
    """
    
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        
        # Contact detector
        self.contact_detector = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, 1),
            nn.Sigmoid(),
        )
        
        # Causal plausibility given contact
        self.causal_given_contact = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim // 2),
            nn.GELU(),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid(),
        )
    
    def forward(
        self,
        cause_state: torch.Tensor,
        effect_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Assess causal plausibility based on contact.
        
        Returns:
            - had_contact: [B] were they in contact?
            - causal_plausible: [B] is causation plausible?
        """
        combined = torch.cat([cause_state, effect_state], dim=-1)
        
        had_contact = self.contact_detector(combined).squeeze(-1)
        causal_plausible = self.causal_given_contact(combined).squeeze(-1)
        
        # Causation requires contact (mostly)
        final_plausibility = causal_plausible * (0.3 + 0.7 * had_contact)
        
        return had_contact > 0.5, final_plausibility


class IntuitivePhysics(nn.Module):
    """
    Combined intuitive physics module.
    
    Encodes fundamental physics intuitions that humans seem to have innately:
    - Gravity: Things fall
    - Solidity: Things don't pass through each other
    - Support: Things need support
    - Contact: Causation needs contact
    
    These are encoded as soft priors, not hard rules.
    Experience can override them (e.g., learning about magnets).
    """
    
    def __init__(self, feature_dim: int = 256) -> None:
        super().__init__()
        
        self.gravity = GravityPrior(feature_dim)
        self.solidity = SolidityPrior(feature_dim)
        self.contact_causality = ContactCausalityPrior(feature_dim)
        
        # Physics violation detector (learned)
        self.violation_detector = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, len(PhysicsLaw)),
            nn.Sigmoid(),
        )
    
    def check_gravity(
        self,
        object_state: torch.Tensor,
        actual_motion: torch.Tensor,
    ) -> PhysicsViolation:
        """Check if motion violates gravity expectation."""
        is_supported, expected_motion = self.gravity(object_state)
        
        # Compare expected vs actual
        motion_error = (actual_motion - expected_motion).norm(dim=-1)
        violation_severity = torch.clamp(motion_error / 2.0, 0, 1).mean().item()
        
        if violation_severity > 0.5:
            return PhysicsViolation(
                law=PhysicsLaw.GRAVITY,
                severity=violation_severity,
                description=f"Object moved unexpectedly (severity: {violation_severity:.2f})"
            )
        return None
    
    def check_solidity(
        self,
        object1_state: torch.Tensor,
        object2_state: torch.Tensor,
    ) -> PhysicsViolation:
        """Check if two objects violate solidity."""
        violation_prob = self.solidity(object1_state, object2_state)
        severity = violation_prob.mean().item()
        
        if severity > 0.5:
            return PhysicsViolation(
                law=PhysicsLaw.SOLIDITY,
                severity=severity,
                description=f"Objects appear to pass through each other"
            )
        return None
    
    def check_all(
        self,
        state_before: torch.Tensor,
        state_after: torch.Tensor,
    ) -> List[PhysicsViolation]:
        """Check all physics laws."""
        combined = torch.cat([state_before, state_after], dim=-1)
        violation_probs = self.violation_detector(combined)
        
        violations = []
        
        if violation_probs.dim() > 1:
            violation_probs = violation_probs[0]
        
        for i, law in enumerate(PhysicsLaw):
            if violation_probs[i] > 0.5:
                violations.append(PhysicsViolation(
                    law=law,
                    severity=violation_probs[i].item(),
                    description=f"{law.value} violation detected"
                ))
        
        return violations
    
    def predict_physics_outcome(
        self,
        initial_state: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict what physics says should happen.
        
        Returns expected outcomes based on physics priors.
        """
        is_supported, expected_gravity_motion = self.gravity(initial_state)
        
        return {
            'is_supported': is_supported,
            'expected_motion': expected_gravity_motion,
        }
    
    def physics_surprise(
        self,
        predicted_state: torch.Tensor,
        actual_state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute how surprising the actual outcome is given physics priors.
        
        High surprise = physics violation = worth paying attention to.
        """
        diff = (predicted_state - actual_state).norm(dim=-1)
        # Normalize to [0, 1] surprise
        surprise = 1 - torch.exp(-diff)
        return surprise
