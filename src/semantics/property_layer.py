"""
Property Layer - Extract semantic properties from world state.

This layer transforms raw perceptual features into meaningful properties
that humans naturally perceive:
- Hardness (from sound, visual deformation)
- Weight (from acceleration under force)
- Size (from visual extent)
- Animacy (from self-propelled motion)
- Texture (from visual patterns)
- Temperature (would need tactile, approximated from context)

Key insight: A rock is understood as "hard, gray, heavy, small" not as
a pixel pattern. This is how semantic understanding emerges from perception.

ARCHITECTURAL UPDATE (Peer Review):
The fixed NamedTuple PropertyVector has been supplemented with
DynamicPropertyBank which uses slot attention to enable:
1. Open-ended property discovery (new properties like "stickiness")
2. Free slots activated by prediction error
3. Post-hoc grounding of discovered properties
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Slot Attention for Dynamic Properties
# ============================================================================

class SlotAttention(nn.Module):
    """
    Slot Attention mechanism for property extraction.
    
    Inspired by Locatello et al. (2020) "Object-Centric Learning with
    Slot Attention", adapted for property discovery.
    
    Each slot represents a property. Known slots (0-8) are initialized
    with priors for hardness, weight, etc. Free slots (9-N) can discover
    new properties through learning.
    """
    
    def __init__(
        self,
        num_slots: int,
        slot_dim: int,
        input_dim: int,
        num_iterations: int = 3,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.num_iterations = num_iterations
        
        # Slot initialization (learned)
        self.slots_mu = nn.Parameter(torch.randn(1, num_slots, slot_dim))
        self.slots_sigma = nn.Parameter(torch.ones(1, num_slots, slot_dim) * 0.1)
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, slot_dim)
        
        # Attention
        self.k_proj = nn.Linear(slot_dim, slot_dim)
        self.q_proj = nn.Linear(slot_dim, slot_dim)
        self.v_proj = nn.Linear(slot_dim, slot_dim)
        
        # GRU for iterative refinement
        self.gru = nn.GRUCell(slot_dim, slot_dim)
        
        # MLP for slot update
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, slot_dim),
        )
        
        # Layer norms
        self.norm_input = nn.LayerNorm(slot_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.norm_mlp = nn.LayerNorm(slot_dim)
        
        self.scale = slot_dim ** -0.5
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Apply slot attention to extract properties.
        
        Args:
            inputs: Input features [B, N, input_dim] or [B, input_dim]
            
        Returns:
            slots: Property slots [B, num_slots, slot_dim]
        """
        B = inputs.shape[0]
        
        # Handle 2D input (no sequence dimension)
        if inputs.dim() == 2:
            inputs = inputs.unsqueeze(1)  # [B, 1, D]
        
        # Project inputs
        inputs = self.input_proj(inputs)
        inputs = self.norm_input(inputs)
        
        # Initialize slots
        slots = self.slots_mu.expand(B, -1, -1)
        slots = slots + self.slots_sigma * torch.randn_like(slots)
        
        # Iterative attention
        for _ in range(self.num_iterations):
            slots_prev = slots
            slots = self.norm_slots(slots)
            
            # Attention
            k = self.k_proj(inputs)  # [B, N, D]
            v = self.v_proj(inputs)  # [B, N, D]
            q = self.q_proj(slots)   # [B, num_slots, D]
            
            # Scaled dot-product attention
            attn = torch.einsum('bnd,bsd->bns', k, q) * self.scale  # [B, N, num_slots]
            attn = F.softmax(attn, dim=-1)
            attn = attn + 1e-8  # For numerical stability
            
            # Weighted mean
            attn_weights = attn / attn.sum(dim=1, keepdim=True)
            updates = torch.einsum('bns,bnd->bsd', attn_weights, v)
            
            # GRU update
            slots = self.gru(
                updates.reshape(-1, self.slot_dim),
                slots_prev.reshape(-1, self.slot_dim),
            ).reshape(B, self.num_slots, self.slot_dim)
            
            # MLP update
            slots = slots + self.mlp(self.norm_mlp(slots))
        
        return slots


class RobustSlotAttention(nn.Module):
    """
    Slot Attention with reconstruction-based OOD detection.
    
    Addresses adversarial robustness concern from peer review:
    Neuro-symbolic systems can fail catastrophically when the neural
    frontend is attacked. Dynamic slots make this worse (adversarial
    perturbation could create spurious slots).
    
    Solution: Don't populate symbolic layer if input can't be reconstructed.
    
    If reconstruction_error > threshold:
        return slots, confidence="uncertain"
    
    This is similar to how VAEs detect out-of-distribution inputs.
    """
    
    def __init__(
        self,
        num_slots: int,
        slot_dim: int,
        input_dim: int,
        num_iterations: int = 3,
        hidden_dim: int = 128,
        reconstruction_threshold: float = 0.1,
    ) -> None:
        super().__init__()
        
        self.reconstruction_threshold = reconstruction_threshold
        
        # Base slot attention
        self.slot_attention = SlotAttention(
            num_slots=num_slots,
            slot_dim=slot_dim,
            input_dim=input_dim,
            num_iterations=num_iterations,
            hidden_dim=hidden_dim,
        )
        
        # Decoder for reconstruction
        self.decoder = nn.Sequential(
            nn.Linear(slot_dim * num_slots, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )
        
        # Confidence estimator
        self.confidence_net = nn.Sequential(
            nn.Linear(slot_dim * num_slots + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.input_dim = input_dim
    
    def forward(
        self,
        inputs: torch.Tensor,
        return_confidence: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], str]:
        """
        Apply slot attention with OOD detection.
        
        Args:
            inputs: Input features [B, N, input_dim] or [B, input_dim]
            return_confidence: Whether to compute confidence
            
        Returns:
            - slots: [B, num_slots, slot_dim]
            - confidence: [B] confidence score (if return_confidence)
            - status: "certain" or "uncertain"
        """
        # Handle 2D input
        if inputs.dim() == 2:
            inputs_3d = inputs.unsqueeze(1)
        else:
            inputs_3d = inputs
        
        B = inputs_3d.shape[0]
        
        # Get slots
        slots = self.slot_attention(inputs_3d)  # [B, num_slots, slot_dim]
        
        if not return_confidence:
            return slots, None, "certain"
        
        # Reconstruct input from slots
        slots_flat = slots.reshape(B, -1)  # [B, num_slots * slot_dim]
        reconstruction = self.decoder(slots_flat)  # [B, input_dim]
        
        # Compute reconstruction error
        if inputs.dim() == 2:
            target = inputs
        else:
            target = inputs.mean(dim=1)  # Average over sequence
        
        recon_error = F.mse_loss(reconstruction, target, reduction='none')
        recon_error = recon_error.mean(dim=-1, keepdim=True)  # [B, 1]
        
        # Compute confidence
        confidence_input = torch.cat([slots_flat, recon_error], dim=-1)
        confidence = self.confidence_net(confidence_input).squeeze(-1)  # [B]
        
        # Determine status based on reconstruction error
        avg_error = recon_error.mean().item()
        if avg_error > self.reconstruction_threshold:
            status = "uncertain"
        else:
            status = "certain"
        
        return slots, confidence, status
    
    def is_out_of_distribution(
        self,
        inputs: torch.Tensor,
        threshold: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Check if inputs are out of distribution.
        
        Args:
            inputs: Input features
            threshold: Optional custom threshold
            
        Returns:
            - is_ood: [B] boolean tensor
            - reconstruction_error: [B] error values
        """
        threshold = threshold or self.reconstruction_threshold
        
        # Handle 2D input
        if inputs.dim() == 2:
            inputs_3d = inputs.unsqueeze(1)
        else:
            inputs_3d = inputs
        
        B = inputs_3d.shape[0]
        
        # Get slots and reconstruction
        slots = self.slot_attention(inputs_3d)
        slots_flat = slots.reshape(B, -1)
        reconstruction = self.decoder(slots_flat)
        
        # Compute error
        if inputs.dim() == 2:
            target = inputs
        else:
            target = inputs.mean(dim=1)
        
        recon_error = F.mse_loss(reconstruction, target, reduction='none')
        recon_error = recon_error.mean(dim=-1)  # [B]
        
        is_ood = recon_error > threshold
        
        return is_ood, recon_error
    
    def get_reconstruction_quality(
        self,
        inputs: torch.Tensor,
    ) -> Dict[str, Any]:
        """Get detailed reconstruction quality metrics."""
        slots, confidence, status = self.forward(inputs, return_confidence=True)
        is_ood, recon_error = self.is_out_of_distribution(inputs)
        
        return {
            'status': status,
            'confidence': confidence.mean().item() if confidence is not None else 0.0,
            'reconstruction_error': recon_error.mean().item(),
            'is_ood_fraction': is_ood.float().mean().item(),
            'threshold': self.reconstruction_threshold,
        }


class DynamicPropertyBank(nn.Module):
    """
    Dynamic property representation with discoverable slots.
    
    Solves the "Ontology Bottleneck" identified in peer review:
    - Fixed NamedTuple cannot discover new properties
    - DynamicPropertyBank reserves free slots for discovery
    
    Architecture:
    - Slots 0-8: Initialized with known priors (hardness, weight, etc.)
    - Slots 9-N: Free slots, initialized near zero, activated by curiosity
    
    Grounding Protocol:
    - Post-hoc: After training, humans/LLM name activated free slots
    - Online: If language available during babbling, contrastive alignment
    
    References:
    - Locatello et al. (2020). Slot Attention for Object-Centric Learning.
    """
    
    # Names for known property slots
    KNOWN_PROPERTY_NAMES = [
        "hardness", "weight", "size", "animacy", "rigidity",
        "transparency", "roughness", "temperature", "containment"
    ]
    
    def __init__(
        self,
        input_dim: int = 256,
        num_slots: int = 32,
        slot_dim: int = 64,
        num_known: int = 9,
    ) -> None:
        super().__init__()
        
        self.input_dim = input_dim
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.num_known = num_known
        self.num_free = num_slots - num_known
        
        # Known property prototypes (initialized, learnable)
        # These correspond to hardness, weight, size, etc.
        self.known_prototypes = nn.Parameter(
            torch.randn(num_known, slot_dim) * 0.1
        )
        
        # Free slots for discovering new properties
        # IMPLEMENTATION NOTE (Slot Attention Collapse Trap):
        # Initialize with ORTHOGONAL noise, not random Gaussian.
        # This prevents all slots from collapsing to the same value.
        free_slot_init = torch.zeros(self.num_free, slot_dim)
        if self.num_free > 0:
            # Use orthogonal initialization for diversity
            nn.init.orthogonal_(free_slot_init)
            free_slot_init *= 0.01  # Scale down but keep orthogonality
        self.free_slots = nn.Parameter(free_slot_init)
        
        # Slot attention for property extraction
        self.slot_attention = SlotAttention(
            num_slots=num_slots,
            slot_dim=slot_dim,
            input_dim=input_dim,
        )
        
        # Property value heads (one per known property)
        self.property_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(slot_dim, slot_dim // 2),
                nn.GELU(),
                nn.Linear(slot_dim // 2, 1),
                nn.Sigmoid(),
            )
            for _ in range(num_known)
        ])
        
        # Free slot activation tracking
        self.free_slot_activations = nn.Parameter(
            torch.zeros(self.num_free), requires_grad=False
        )
        
        # Slot names (known + discovered)
        self.slot_names: Dict[int, str] = {
            i: name for i, name in enumerate(self.KNOWN_PROPERTY_NAMES)
        }
        
        # Text encoder for online grounding (optional)
        self.text_encoder: Optional[nn.Module] = None
        self.grounding_loss = torch.tensor(0.0)
    
    def forward(
        self,
        world_state: torch.Tensor,
        utterance_embedding: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Extract properties using slot attention.
        
        Args:
            world_state: World state features [B, input_dim]
            utterance_embedding: Optional language embedding for grounding
            
        Returns:
            - property_values: [B, num_known] known property values
            - diagnostics: Dict with slot activations, free slot usage, etc.
        """
        B = world_state.shape[0]
        
        # Get all slot prototypes
        all_slots = torch.cat([self.known_prototypes, self.free_slots], dim=0)
        
        # Apply slot attention
        slots = self.slot_attention(world_state)  # [B, num_slots, slot_dim]
        
        # Extract known property values
        property_values = []
        for i, head in enumerate(self.property_heads):
            value = head(slots[:, i, :])  # [B, 1]
            property_values.append(value.squeeze(-1))
        
        property_values = torch.stack(property_values, dim=-1)  # [B, num_known]
        
        # Track free slot activations
        free_slot_features = slots[:, self.num_known:, :]  # [B, num_free, slot_dim]
        free_activations = free_slot_features.norm(dim=-1).mean(dim=0)  # [num_free]
        
        # Update activation tracking (EMA)
        with torch.no_grad():
            self.free_slot_activations.data = (
                0.99 * self.free_slot_activations.data +
                0.01 * free_activations
            )
        
        # Online grounding loss (if language available)
        if utterance_embedding is not None and self.training:
            self.grounding_loss = self._compute_grounding_loss(
                slots, utterance_embedding
            )
        
        diagnostics = {
            'slots': slots,
            'property_values': property_values,
            'free_slot_activations': self.free_slot_activations.clone(),
            'active_free_slots': (self.free_slot_activations > 0.1).sum().item(),
            'grounding_loss': self.grounding_loss.item() if isinstance(self.grounding_loss, torch.Tensor) else 0.0,
        }
        
        return property_values, diagnostics
    
    def _compute_grounding_loss(
        self,
        slots: torch.Tensor,
        text_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute contrastive grounding loss.
        
        Aligns most active slot with language description.
        """
        # Compute slot-text similarity
        slot_features = slots.mean(dim=0)  # [num_slots, slot_dim]
        
        # Project to same space if dimensions differ
        if slot_features.shape[-1] != text_embedding.shape[-1]:
            return torch.tensor(0.0, device=slots.device)
        
        similarities = F.cosine_similarity(
            slot_features.unsqueeze(0),
            text_embedding.unsqueeze(1),
            dim=-1
        )
        
        # Maximize alignment for most active slot
        max_sim = similarities.max(dim=-1).values
        return -max_sim.mean()  # Negative for loss
    
    def activate_free_slot(
        self,
        slot_idx: int,
        initial_embedding: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Activate a free slot for a newly discovered property.
        
        Called by curiosity drive when prediction error suggests
        existing properties don't explain observation.
        """
        if slot_idx >= self.num_free:
            return
        
        if initial_embedding is not None:
            with torch.no_grad():
                self.free_slots.data[slot_idx] = initial_embedding
        else:
            # Initialize with small random values
            with torch.no_grad():
                self.free_slots.data[slot_idx] = torch.randn(self.slot_dim) * 0.1
    
    def ground_free_slot(
        self,
        slot_idx: int,
        name: str,
        examples: Optional[List[torch.Tensor]] = None,
    ) -> None:
        """
        Ground a free slot with a name (post-hoc labeling).
        
        After training, humans or LLM examine what activates the slot
        and provide a descriptive name.
        
        Args:
            slot_idx: Index of free slot (0-indexed within free slots)
            name: Descriptive name (e.g., "stickiness", "elasticity")
            examples: Optional example inputs that activate this slot
        """
        actual_idx = self.num_known + slot_idx
        self.slot_names[actual_idx] = name
    
    def get_active_free_slots(self, threshold: float = 0.1) -> List[int]:
        """Get indices of free slots that have been activated."""
        active = (self.free_slot_activations > threshold).nonzero(as_tuple=True)[0]
        return active.tolist()
    
    def get_property_summary(self) -> Dict[str, Any]:
        """Get summary of property bank state."""
        return {
            'num_known': self.num_known,
            'num_free': self.num_free,
            'active_free_slots': self.get_active_free_slots(),
            'slot_names': dict(self.slot_names),
            'free_slot_activations': self.free_slot_activations.tolist(),
        }
    
    def to_property_vector(self, values: torch.Tensor) -> 'PropertyVector':
        """Convert dynamic property values to legacy PropertyVector."""
        if values.shape[-1] < 9:
            # Pad if needed
            values = F.pad(values, (0, 9 - values.shape[-1]))
        
        return PropertyVector.from_tensor(values[..., :9])


class PropertyVector(NamedTuple):
    """
    Semantic property vector for an object/region.
    
    All properties are continuous values in [0, 1] representing
    the degree to which the property is present.
    """
    hardness: torch.Tensor      # 0=soft, 1=hard
    weight: torch.Tensor        # 0=light, 1=heavy (relative to size)
    size: torch.Tensor          # 0=tiny, 1=large
    animacy: torch.Tensor       # 0=inanimate, 1=animate (agent)
    rigidity: torch.Tensor      # 0=flexible, 1=rigid
    transparency: torch.Tensor  # 0=opaque, 1=transparent
    roughness: torch.Tensor     # 0=smooth, 1=rough
    temperature: torch.Tensor   # 0=cold, 0.5=neutral, 1=hot
    containment: torch.Tensor   # 0=solid, 1=can contain things
    
    def to_tensor(self) -> torch.Tensor:
        """Convert to single tensor [B, 9]."""
        return torch.stack([
            self.hardness, self.weight, self.size, self.animacy,
            self.rigidity, self.transparency, self.roughness,
            self.temperature, self.containment
        ], dim=-1)
    
    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> 'PropertyVector':
        """Create from tensor [B, 9]."""
        return cls(
            hardness=tensor[..., 0],
            weight=tensor[..., 1],
            size=tensor[..., 2],
            animacy=tensor[..., 3],
            rigidity=tensor[..., 4],
            transparency=tensor[..., 5],
            roughness=tensor[..., 6],
            temperature=tensor[..., 7],
            containment=tensor[..., 8],
        )
    
    @classmethod
    def num_properties(cls) -> int:
        return 9


@dataclass
class PropertyConfig:
    """Configuration for property layer."""
    world_state_dim: int = 256
    audio_dim: int = 256
    proprio_dim: int = 256
    hidden_dim: int = 512
    num_properties: int = 9
    use_audio_for_hardness: bool = True
    use_proprio_for_weight: bool = True
    use_motion_for_animacy: bool = True


class PropertyHead(nn.Module):
    """
    Single property extraction head.
    
    Maps multi-modal features to a single property value in [0, 1].
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),  # Output in [0, 1]
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Extract property value.
        
        Args:
            features: Multi-modal features [B, D]
            
        Returns:
            Property value [B] in [0, 1]
        """
        return self.network(features).squeeze(-1)


class HardnessExtractor(nn.Module):
    """
    Extract hardness from audio and visual features.
    
    Hardness correlates with:
    - Audio: High-frequency sounds when struck (hard things clink)
    - Visual: No deformation under force
    """
    
    def __init__(self, visual_dim: int, audio_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        
        # Audio pathway (primary for hardness)
        self.audio_encoder = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )
        
        # Visual pathway (deformation detection)
        self.visual_encoder = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )
        
        # Fusion and output
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        
    def forward(
        self,
        visual_features: torch.Tensor,
        audio_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extract hardness property.
        
        High-frequency audio = hard
        No visual deformation = hard
        """
        visual_enc = self.visual_encoder(visual_features)
        
        if audio_features is not None:
            audio_enc = self.audio_encoder(audio_features)
            combined = torch.cat([visual_enc, audio_enc], dim=-1)
        else:
            # Use only visual if no audio
            combined = torch.cat([visual_enc, torch.zeros_like(visual_enc)], dim=-1)
        
        return self.fusion(combined).squeeze(-1)


class WeightExtractor(nn.Module):
    """
    Extract weight (relative to size) from proprioceptive and visual features.
    
    Weight correlates with:
    - Proprioception: Force required to move / acceleration achieved
    - Visual: Dense appearance, material type
    """
    
    def __init__(self, visual_dim: int, proprio_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        
        # Proprioception pathway (primary for weight)
        self.proprio_encoder = nn.Sequential(
            nn.Linear(proprio_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )
        
        # Visual pathway (material/density estimation)
        self.visual_encoder = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
    
    def forward(
        self,
        visual_features: torch.Tensor,
        proprio_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Extract weight property."""
        visual_enc = self.visual_encoder(visual_features)
        
        if proprio_features is not None:
            proprio_enc = self.proprio_encoder(proprio_features)
            combined = torch.cat([visual_enc, proprio_enc], dim=-1)
        else:
            combined = torch.cat([visual_enc, torch.zeros_like(visual_enc)], dim=-1)
        
        return self.fusion(combined).squeeze(-1)


class AnimacyDetector(nn.Module):
    """
    Detect animacy (is this an agent that moves on its own?).
    
    Animacy correlates with:
    - Self-propelled motion (moves without external force)
    - Biological motion patterns
    - Goal-directed behavior
    
    This is partially innate in humans - babies prefer looking at
    biological motion from birth.
    """
    
    def __init__(self, feature_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        
        # Motion analysis
        self.motion_encoder = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),  # Current + previous state
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Biological motion detector (simplified)
        self.bio_motion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        
        # Self-propulsion detector
        self.self_propelled = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        
        # Combine
        self.combine = nn.Linear(2, 1)
    
    def forward(
        self,
        current_state: torch.Tensor,
        previous_state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Detect animacy from state and motion.
        
        Args:
            current_state: Current world state [B, D]
            previous_state: Previous world state [B, D] (for motion)
            
        Returns:
            Animacy score [B] in [0, 1]
        """
        if previous_state is None:
            previous_state = current_state
        
        # Encode motion
        motion_input = torch.cat([current_state, previous_state], dim=-1)
        motion_features = self.motion_encoder(motion_input)
        
        # Detect biological motion and self-propulsion
        bio_score = self.bio_motion(motion_features)
        self_prop_score = self.self_propelled(motion_features)
        
        # Combine
        combined = torch.cat([bio_score, self_prop_score], dim=-1)
        animacy = torch.sigmoid(self.combine(combined))
        
        return animacy.squeeze(-1)


class PropertyLayer(nn.Module):
    """
    Layer 1: Semantic Property Extraction.
    
    Transforms multi-modal world state into semantic properties.
    This is how "rock" becomes "hard, gray, heavy, small" instead of
    just a pixel pattern.
    
    Properties are grounded in perception:
    - Hardness: From sound (high freq = hard) and visual (no deformation)
    - Weight: From proprioception (force/acceleration) and visual (density)
    - Animacy: From motion (self-propelled = agent)
    - Size: From visual extent relative to scene
    - etc.
    """
    
    def __init__(self, config: PropertyConfig) -> None:
        super().__init__()
        
        self.config = config
        
        # Multi-modal fusion for property extraction
        total_input_dim = config.world_state_dim
        if config.use_audio_for_hardness:
            total_input_dim += config.audio_dim
        if config.use_proprio_for_weight:
            total_input_dim += config.proprio_dim
        
        # Specialized extractors for key properties
        self.hardness_extractor = HardnessExtractor(
            visual_dim=config.world_state_dim,
            audio_dim=config.audio_dim,
            hidden_dim=config.hidden_dim // 2,
        )
        
        self.weight_extractor = WeightExtractor(
            visual_dim=config.world_state_dim,
            proprio_dim=config.proprio_dim,
            hidden_dim=config.hidden_dim // 2,
        )
        
        self.animacy_detector = AnimacyDetector(
            feature_dim=config.world_state_dim,
            hidden_dim=config.hidden_dim // 2,
        )
        
        # Generic heads for other properties
        self.size_head = PropertyHead(config.world_state_dim, config.hidden_dim // 2)
        self.rigidity_head = PropertyHead(config.world_state_dim, config.hidden_dim // 2)
        self.transparency_head = PropertyHead(config.world_state_dim, config.hidden_dim // 2)
        self.roughness_head = PropertyHead(config.world_state_dim, config.hidden_dim // 2)
        self.temperature_head = PropertyHead(config.world_state_dim, config.hidden_dim // 2)
        self.containment_head = PropertyHead(config.world_state_dim, config.hidden_dim // 2)
        
        # Output projection (optional - for combined property vector)
        self.output_proj = nn.Linear(
            PropertyVector.num_properties(),
            config.hidden_dim
        )
    
    def forward(
        self,
        world_state: torch.Tensor,
        audio_features: Optional[torch.Tensor] = None,
        proprio_features: Optional[torch.Tensor] = None,
        previous_state: Optional[torch.Tensor] = None,
    ) -> Tuple[PropertyVector, torch.Tensor]:
        """
        Extract semantic properties from world state.
        
        Args:
            world_state: World state from Layer 0 [B, D]
            audio_features: Audio features [B, audio_dim] (optional)
            proprio_features: Proprioception [B, proprio_dim] (optional)
            previous_state: Previous world state for motion [B, D] (optional)
            
        Returns:
            Tuple of:
            - PropertyVector with all properties
            - Property embedding [B, hidden_dim]
        """
        # Extract specialized properties
        hardness = self.hardness_extractor(world_state, audio_features)
        weight = self.weight_extractor(world_state, proprio_features)
        animacy = self.animacy_detector(world_state, previous_state)
        
        # Extract generic properties
        size = self.size_head(world_state)
        rigidity = self.rigidity_head(world_state)
        transparency = self.transparency_head(world_state)
        roughness = self.roughness_head(world_state)
        temperature = self.temperature_head(world_state)
        containment = self.containment_head(world_state)
        
        # Create property vector
        properties = PropertyVector(
            hardness=hardness,
            weight=weight,
            size=size,
            animacy=animacy,
            rigidity=rigidity,
            transparency=transparency,
            roughness=roughness,
            temperature=temperature,
            containment=containment,
        )
        
        # Create embedding
        prop_tensor = properties.to_tensor()
        embedding = self.output_proj(prop_tensor)
        
        return properties, embedding
    
    def property_similarity(
        self,
        props1: PropertyVector,
        props2: PropertyVector,
    ) -> torch.Tensor:
        """
        Compute similarity between two property vectors.
        
        Used for concept matching: "Is this thing similar to a rock?"
        """
        t1 = props1.to_tensor()
        t2 = props2.to_tensor()
        
        # Cosine similarity
        return F.cosine_similarity(t1, t2, dim=-1)
    
    def describe_properties(self, properties: PropertyVector) -> Dict[str, str]:
        """
        Convert property vector to human-readable description.
        
        Useful for debugging and language grounding.
        """
        def level(val: float) -> str:
            if val < 0.3:
                return "low"
            elif val < 0.7:
                return "medium"
            else:
                return "high"
        
        # Get first element if batched
        h = properties.hardness[0].item() if properties.hardness.dim() > 0 else properties.hardness.item()
        w = properties.weight[0].item() if properties.weight.dim() > 0 else properties.weight.item()
        s = properties.size[0].item() if properties.size.dim() > 0 else properties.size.item()
        a = properties.animacy[0].item() if properties.animacy.dim() > 0 else properties.animacy.item()
        
        return {
            "hardness": f"{level(h)} ({h:.2f})",
            "weight": f"{level(w)} ({w:.2f})",
            "size": f"{level(s)} ({s:.2f})",
            "animacy": "animate" if a > 0.5 else "inanimate",
        }
