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
    Innate understanding of gravity (DEPRECATED - use AdaptivePhysicsPrior).
    
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


class AdaptivePhysicsPrior(nn.Module):
    """
    Physics prior with learnable correction - the core innovation.
    
    Key insight from peer review: Hard-coded physics (gravity = -9.8) breaks
    on exceptions like balloons, magnets, or zero-gravity. But pure learning
    requires too much data.
    
    Solution: Bayesian initialization + learned residual
    
    output = prior_weight * prior_prediction + (1 - prior_weight) * correction
    
    Where:
    - prior_prediction: The innate physics expectation (gravity, etc.)
    - correction: A learned network that handles exceptions
    - prior_weight: Starts at 0.9, can adapt, but floors at 0.3 (critical period)
    
    The critical period floor (0.3) ensures the system never completely
    "forgets" physics, mimicking biological brain plasticity.
    
    References:
    - Baillargeon, R. (2004). Infants' physical world.
    - Smith, L., & Thelen, E. (2003). Development as a dynamic system.
    """
    
    def __init__(
        self,
        feature_dim: int,
        prior_gravity: float = -9.8,
        initial_prior_weight: float = 0.9,
        min_prior_weight: float = 0.3,  # Critical period floor
    ) -> None:
        super().__init__()
        
        self.feature_dim = feature_dim
        self.prior_gravity = prior_gravity
        self.min_prior_weight = min_prior_weight
        
        # Learnable prior weight (starts trusting the prior)
        # Use unconstrained parameter, apply soft constraint in forward
        self._prior_weight = nn.Parameter(
            torch.tensor(initial_prior_weight - min_prior_weight)
        )
        
        # Support detector (is object supported?)
        self.support_detector = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.LayerNorm(feature_dim // 2),
            nn.GELU(),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid(),
        )
        
        # Correction network - learns exceptions to physics priors
        # E.g., "balloons go up", "magnets attract/repel"
        self.correction_net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Linear(feature_dim // 2, 3),  # dx, dy, dz correction
        )
        
        # Context encoder for scene understanding
        self.context_encoder = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Linear(feature_dim // 2, feature_dim // 4),
        )
    
    @property
    def prior_weight(self) -> torch.Tensor:
        """
        Get effective prior weight with soft constraint.
        
        CRITICAL: Use softplus to ensure gradients flow even at boundary.
        This was identified in peer review as a potential bug.
        
        effective_weight = min_weight + softplus(raw_weight)
        
        This guarantees:
        1. Weight is always >= min_prior_weight (critical period)
        2. Gradients flow even when weight is near the floor
        """
        return self.min_prior_weight + F.softplus(self._prior_weight)
    
    def forward(
        self,
        object_state: torch.Tensor,
        scene_context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Predict physics-based motion with adaptive prior.
        
        Args:
            object_state: Object features [B, feature_dim]
            scene_context: Optional scene context [B, feature_dim]
            
        Returns:
            - is_supported: [B] support prediction
            - predicted_motion: [B, 3] combined prior + correction
            - diagnostics: Dict with prior_weight, correction, etc.
        """
        batch_size = object_state.shape[0]
        device = object_state.device
        
        # Detect support
        is_supported = self.support_detector(object_state).squeeze(-1)
        
        # Prior prediction: gravity pulls unsupported objects down
        # Prior motion is gravity-based
        gravity_vec = torch.tensor(
            [0.0, self.prior_gravity, 0.0], 
            device=device
        ).unsqueeze(0).expand(batch_size, -1)
        
        # Supported objects don't fall
        support_mask = (1 - is_supported).unsqueeze(-1)
        prior_motion = gravity_vec * support_mask
        
        # Correction: learned exceptions
        correction = self.correction_net(object_state)
        
        # Blend with adaptive weight
        # prior_weight = 0.9 -> mostly trust physics
        # prior_weight = 0.3 -> learned corrections dominate (but physics not forgotten)
        effective_weight = self.prior_weight
        
        # Ensure weight is properly shaped for broadcasting
        if effective_weight.dim() == 0:
            effective_weight = effective_weight.unsqueeze(0)
        
        predicted_motion = (
            effective_weight * prior_motion + 
            (1 - effective_weight) * correction
        )
        
        # Diagnostics for analysis
        diagnostics = {
            'prior_weight': self.prior_weight.detach(),
            'raw_prior_weight': self._prior_weight.detach(),
            'prior_motion': prior_motion,
            'correction': correction,
            'is_supported': is_supported,
        }
        
        return is_supported > 0.5, predicted_motion, diagnostics
    
    def critical_period_loss(self) -> torch.Tensor:
        """
        Regularization to respect critical period.
        
        Encourages the model to not deviate too far from prior trust,
        especially early in training.
        
        Add this to the main loss: loss += 0.01 * model.critical_period_loss()
        """
        # Penalty for prior_weight going below initial value too quickly
        # This is already enforced by softplus, but we can add soft encouragement
        # to stay closer to initial trust
        deviation = F.relu(0.6 - self.prior_weight)  # Penalty below 0.6
        return deviation * 10.0
    
    def get_prior_statistics(self) -> Dict[str, float]:
        """Get current prior weight statistics."""
        return {
            'prior_weight': self.prior_weight.item(),
            'raw_weight': self._prior_weight.item(),
            'min_weight': self.min_prior_weight,
            'correction_magnitude': 0.0,  # Computed during forward
        }


class AdaptiveSolidityPrior(nn.Module):
    """
    Adaptive solidity prior - objects don't pass through each other.
    
    Uses same residual architecture as AdaptivePhysicsPrior.
    """
    
    def __init__(
        self,
        feature_dim: int,
        initial_prior_weight: float = 0.9,
        min_prior_weight: float = 0.3,
    ) -> None:
        super().__init__()
        
        self.min_prior_weight = min_prior_weight
        self._prior_weight = nn.Parameter(
            torch.tensor(initial_prior_weight - min_prior_weight)
        )
        
        # Prior: overlap detection
        self.overlap_detector = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, 1),
            nn.Sigmoid(),
        )
        
        # Correction: learned exceptions (e.g., permeable surfaces)
        self.correction_net = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, 1),
            nn.Sigmoid(),
        )
    
    @property
    def prior_weight(self) -> torch.Tensor:
        return self.min_prior_weight + F.softplus(self._prior_weight)
    
    def forward(
        self,
        object1_state: torch.Tensor,
        object2_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Detect if two objects violate solidity with adaptive prior."""
        combined = torch.cat([object1_state, object2_state], dim=-1)
        
        # Prior: simple overlap detection
        prior_violation = self.overlap_detector(combined).squeeze(-1)
        
        # Correction: learned exceptions
        correction = self.correction_net(combined).squeeze(-1)
        
        # Blend
        violation_prob = (
            self.prior_weight * prior_violation +
            (1 - self.prior_weight) * correction
        )
        
        diagnostics = {
            'prior_weight': self.prior_weight.detach(),
            'prior_violation': prior_violation,
            'correction': correction,
        }
        
        return violation_prob, diagnostics


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
    Combined intuitive physics module with ADAPTIVE priors.
    
    Key architectural change from peer review:
    - OLD: Hard-coded physics rules that break on exceptions
    - NEW: Learnable priors with residual correction networks
    
    Physics intuitions (gravity, solidity, support, contact) are encoded
    as BIASES, not RULES. The system starts trusting these priors (weight=0.9)
    but can learn to override them for exceptions (balloons, magnets).
    
    The critical period floor (0.3) ensures physics knowledge is never
    completely forgotten, mimicking biological brain plasticity.
    
    References:
    - Spelke, E. S., & Kinzler, K. D. (2007). Core knowledge.
    - Baillargeon, R. (2004). Infants' physical world.
    """
    
    def __init__(
        self,
        feature_dim: int = 256,
        use_adaptive_priors: bool = True,
        initial_prior_weight: float = 0.9,
        min_prior_weight: float = 0.3,
    ) -> None:
        super().__init__()
        
        self.use_adaptive_priors = use_adaptive_priors
        self.feature_dim = feature_dim
        
        # Use adaptive priors by default (the new architecture)
        if use_adaptive_priors:
            self.gravity = AdaptivePhysicsPrior(
                feature_dim,
                initial_prior_weight=initial_prior_weight,
                min_prior_weight=min_prior_weight,
            )
            self.solidity = AdaptiveSolidityPrior(
                feature_dim,
                initial_prior_weight=initial_prior_weight,
                min_prior_weight=min_prior_weight,
            )
        else:
            # Legacy: hard-coded priors (for ablation study)
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
    ) -> Optional[PhysicsViolation]:
        """Check if motion violates gravity expectation."""
        if self.use_adaptive_priors:
            is_supported, expected_motion, diagnostics = self.gravity(object_state)
        else:
            is_supported, expected_motion = self.gravity(object_state)
            diagnostics = {}
        
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
    ) -> Optional[PhysicsViolation]:
        """Check if two objects violate solidity."""
        if self.use_adaptive_priors:
            violation_prob, diagnostics = self.solidity(object1_state, object2_state)
        else:
            violation_prob = self.solidity(object1_state, object2_state)
            diagnostics = {}
        
        severity = violation_prob.mean().item()
        
        if severity > 0.5:
            return PhysicsViolation(
                law=PhysicsLaw.SOLIDITY,
                severity=severity,
                description=f"Objects appear to pass through each other"
            )
        return None
    
    def get_prior_weights(self) -> Dict[str, float]:
        """Get current prior weights for all physics priors."""
        weights = {}
        
        if self.use_adaptive_priors:
            if hasattr(self.gravity, 'prior_weight'):
                weights['gravity'] = self.gravity.prior_weight.item()
            if hasattr(self.solidity, 'prior_weight'):
                weights['solidity'] = self.solidity.prior_weight.item()
        else:
            weights['gravity'] = 1.0  # Fixed priors have implicit weight of 1
            weights['solidity'] = 1.0
            
        return weights
    
    def get_critical_period_loss(self) -> torch.Tensor:
        """
        Get combined critical period regularization loss.
        
        Add to main loss: loss += 0.01 * physics.get_critical_period_loss()
        """
        loss = torch.tensor(0.0)
        
        if self.use_adaptive_priors:
            if hasattr(self.gravity, 'critical_period_loss'):
                loss = loss + self.gravity.critical_period_loss()
                
        return loss
    
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
        With adaptive priors, also returns diagnostic information.
        """
        if self.use_adaptive_priors:
            is_supported, expected_gravity_motion, diagnostics = self.gravity(initial_state)
            return {
                'is_supported': is_supported,
                'expected_motion': expected_gravity_motion,
                'prior_weight': diagnostics.get('prior_weight'),
                'correction': diagnostics.get('correction'),
            }
        else:
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
