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
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F


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
