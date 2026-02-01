"""
Unified World Model.

The complete world model that integrates everything:
- Multi-modal encoders with innate priors
- Cross-modal fusion
- Temporal understanding
- Dynamics prediction (imagination)
- Memory systems
- Few-shot learning

This is the "brain" that perceives, understands, remembers, and imagines.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.encoders.vision_encoder import VisionEncoderWithPriors, VisionEncoderConfig
from src.encoders.audio_encoder import AudioEncoderWithPriors, AudioEncoderConfig
from src.encoders.proprio_encoder import ProprioEncoder, ProprioEncoderConfig
from src.fusion.cross_modal import CrossModalFusion, FusionConfig
from src.world_model.temporal_world_model import TemporalWorldModel, TemporalWorldModelConfig
from src.world_model.enhanced_dynamics import EnhancedDynamicsPredictor, EnhancedDynamicsConfig
from src.memory.dual_memory import DualMemorySystem
from src.learning.meta_learner import MetaLearner, PrototypicalNetworks


@dataclass
class WorldModelConfig:
    """Configuration for the unified world model."""
    # Feature dimensions
    latent_dim: int = 512
    state_dim: int = 256
    action_dim: int = 32
    
    # Vision encoder
    vision: VisionEncoderConfig = field(default_factory=lambda: VisionEncoderConfig(
        input_height=224,
        input_width=224,
        latent_dim=512,
    ))
    
    # Audio encoder
    audio: AudioEncoderConfig = field(default_factory=lambda: AudioEncoderConfig(
        sample_rate=16000,
        n_mels=80,
        latent_dim=256,
        output_dim=512,
    ))
    
    # Proprioception encoder
    proprio: ProprioEncoderConfig = field(default_factory=lambda: ProprioEncoderConfig(
        input_dim=12,
        output_dim=512,
    ))
    
    # Fusion
    fusion: FusionConfig = field(default_factory=lambda: FusionConfig(
        dim=512,
        num_heads=8,
        num_layers=4,
    ))
    
    # Temporal
    temporal: TemporalWorldModelConfig = field(default_factory=lambda: TemporalWorldModelConfig(
        dim=512,
        num_heads=8,
        num_layers=6,
        max_seq_len=64,
        state_dim=256,
    ))
    
    # Dynamics
    dynamics: EnhancedDynamicsConfig = field(default_factory=lambda: EnhancedDynamicsConfig(
        state_dim=256,
        action_dim=32,
        hidden_dim=512,
    ))
    
    # Memory
    max_episodic_memories: int = 10000
    consolidation_threshold: int = 5
    
    # Meta-learning
    meta_inner_lr: float = 0.01
    meta_inner_steps: int = 5
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'WorldModelConfig':
        """Create config from dictionary."""
        return cls(
            latent_dim=config_dict.get('latent_dim', 512),
            state_dim=config_dict.get('state_dim', 256),
            action_dim=config_dict.get('action_dim', 32),
            vision=VisionEncoderConfig(**config_dict.get('vision', {})),
            audio=AudioEncoderConfig(**config_dict.get('audio', {})),
            proprio=ProprioEncoderConfig(**config_dict.get('proprio', {})),
            fusion=FusionConfig(**config_dict.get('fusion', {})),
            temporal=TemporalWorldModelConfig(**config_dict.get('temporal', {})),
            dynamics=EnhancedDynamicsConfig(**config_dict.get('dynamics', {})),
        )


class UnifiedWorldModel(nn.Module):
    """
    The complete world model that integrates everything.
    
    This model can:
    1. Perceive: Encode multi-modal inputs (vision, audio, proprioception)
    2. Understand: Fuse modalities and process temporally
    3. Remember: Store and recall experiences
    4. Imagine: Predict future states given actions
    5. Learn: Adapt to new concepts from few examples
    
    This is the core of the neuro-symbolic cognitive agent.
    """
    
    def __init__(self, config: WorldModelConfig) -> None:
        super().__init__()
        
        self.config = config
        
        # ===== ENCODERS WITH INNATE PRIORS =====
        
        self.vision_encoder = VisionEncoderWithPriors(config.vision)
        self.audio_encoder = AudioEncoderWithPriors(config.audio)
        self.proprio_encoder = ProprioEncoder(config.proprio)
        
        # ===== CROSS-MODAL FUSION =====
        
        self.fusion = CrossModalFusion(config.fusion)
        
        # ===== TEMPORAL PROCESSING =====
        
        self.temporal_model = TemporalWorldModel(config.temporal)
        
        # ===== DYNAMICS (IMAGINATION) =====
        
        self.dynamics = EnhancedDynamicsPredictor(config.dynamics)
        
        # ===== MEMORY =====
        
        # Memory is not a nn.Module, so we don't register it
        self.memory = DualMemorySystem(
            dim=config.state_dim,
            max_episodic=config.max_episodic_memories,
            consolidation_threshold=config.consolidation_threshold,
        )
        
        # ===== META-LEARNING =====
        
        # Create meta-learner for vision encoder
        self.meta_learner = MetaLearner(
            self.vision_encoder,
            inner_lr=config.meta_inner_lr,
            inner_steps=config.meta_inner_steps,
        )
        
        # Prototypical networks for simpler few-shot
        self.proto_nets = PrototypicalNetworks(self.vision_encoder)
    
    def encode_vision(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode visual input.
        
        Args:
            images: RGB images [B, 3, H, W] or [B, T, 3, H, W] for video
            
        Returns:
            Visual features [B, D] or [B, T, D]
        """
        if images.dim() == 5:
            # Video: [B, T, C, H, W]
            B, T, C, H, W = images.shape
            images_flat = images.reshape(B * T, C, H, W)
            features = self.vision_encoder(images_flat)  # [B*T, D, H', W']
            features = features.mean(dim=(2, 3))  # [B*T, D]
            return features.reshape(B, T, -1)  # [B, T, D]
        else:
            # Single image: [B, C, H, W]
            features = self.vision_encoder(images)  # [B, D, H', W']
            return features.mean(dim=(2, 3))  # [B, D]
    
    def encode_audio(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Encode audio input.
        
        Args:
            waveform: Audio waveform [B, T_samples]
            
        Returns:
            Audio features [B, D]
        """
        return self.audio_encoder(waveform)
    
    def encode_proprio(self, body_state: torch.Tensor) -> torch.Tensor:
        """
        Encode proprioceptive input.
        
        Args:
            body_state: Body state [B, 12] or [B, T, 12]
            
        Returns:
            Proprioception features [B, D] or [B, T, D]
        """
        return self.proprio_encoder(body_state)
    
    def encode_multimodal(
        self,
        vision: torch.Tensor,
        audio: torch.Tensor,
        proprio: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Encode and fuse multi-modal inputs.
        
        Args:
            vision: RGB video [B, T, C, H, W] or features [B, T, D]
            audio: Audio waveform [B, T_samples] or features [B, D]
            proprio: Body state [B, T, 12] or features [B, T, D]
            
        Returns:
            Tuple of:
            - Fused features [B, T_total, D]
            - Attention weights (optional)
        """
        # Encode if raw inputs
        if vision.dim() == 5:
            vision = self.encode_vision(vision)  # [B, T, D]
        
        if audio.dim() == 2 and audio.shape[1] > self.config.latent_dim:
            # Likely raw waveform
            audio = self.encode_audio(audio)  # [B, D]
        
        if proprio.dim() == 3 and proprio.shape[2] == 12:
            # Raw body state
            proprio = self.encode_proprio(proprio)  # [B, T, D]
        
        # Ensure all have time dimension
        if vision.dim() == 2:
            vision = vision.unsqueeze(1)
        if audio.dim() == 2:
            T = vision.shape[1]
            audio = audio.unsqueeze(1).expand(-1, T, -1)
        if proprio.dim() == 2:
            proprio = proprio.unsqueeze(1)
        
        # Fuse modalities
        fused, attn = self.fusion(vision, audio, proprio)
        
        return fused, attn
    
    def build_world_state(
        self,
        vision: torch.Tensor,
        audio: torch.Tensor,
        proprio: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build unified world state from multi-modal inputs.
        
        Args:
            vision: Visual input (raw or encoded)
            audio: Audio input (raw or encoded)
            proprio: Proprioception input (raw or encoded)
            
        Returns:
            World state [B, state_dim]
        """
        # Encode and fuse
        fused, _ = self.encode_multimodal(vision, audio, proprio)
        
        # Temporal processing
        world_state, _ = self.temporal_model(fused)
        
        return world_state
    
    def imagine(
        self,
        current_state: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Imagine future states given planned actions.
        
        Args:
            current_state: Current world state [B, state_dim]
            actions: Sequence of actions [B, T, action_dim]
            
        Returns:
            Tuple of:
            - Predicted states [B, T+1, state_dim]
            - Uncertainties [B, T, 1]
        """
        return self.dynamics.predict_trajectory(current_state, actions)
    
    def imagine_single_step(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Imagine single step into future.
        
        Args:
            state: Current state [B, state_dim]
            action: Action [B, action_dim]
            
        Returns:
            Predicted next state and uncertainty
        """
        return self.dynamics(state, action)
    
    def remember(
        self,
        state: torch.Tensor,
        metadata: Dict[str, Any],
    ) -> None:
        """
        Store experience in memory.
        
        Args:
            state: World state to remember [state_dim]
            metadata: Associated information (label, timestamp, etc.)
        """
        self.memory.store(state, metadata)
    
    def recall(
        self,
        query: torch.Tensor,
        mode: str = 'both',
    ) -> Dict[str, Any]:
        """
        Recall from memory.
        
        Args:
            query: Query vector [state_dim]
            mode: 'episodic', 'semantic', or 'both'
            
        Returns:
            Dict with recalled memories
        """
        return self.memory.recall(query, mode)
    
    def learn_new_concept(
        self,
        examples: torch.Tensor,
        concept_name: str,
        use_meta_learning: bool = False,
    ) -> None:
        """
        Learn a new concept from few examples.
        
        Args:
            examples: Example images [K, C, H, W]
            concept_name: Name for the concept
            use_meta_learning: Whether to use MAML adaptation
        """
        # Encode examples
        with torch.no_grad():
            features = self.encode_vision(examples)  # [K, D]
        
        # Store as semantic concept
        self.memory.learn_new_concept(features, concept_name)
    
    def classify_with_memory(
        self,
        query: torch.Tensor,
        top_k: int = 3,
    ) -> List[Tuple[str, float]]:
        """
        Classify query using semantic memory.
        
        Args:
            query: Query image [C, H, W] or features [D]
            top_k: Number of top concepts to return
            
        Returns:
            List of (concept_name, similarity) tuples
        """
        # Encode if image
        if query.dim() == 3:
            query = query.unsqueeze(0)
            query = self.encode_vision(query).squeeze(0)
        
        # Recall from semantic memory
        return self.memory.recall_semantic(query, k=top_k)
    
    def forward(
        self,
        vision: torch.Tensor,
        audio: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass through the world model.
        
        Args:
            vision: Visual input [B, T, C, H, W] or [B, C, H, W]
            audio: Audio input [B, T_samples] (optional)
            proprio: Proprioception [B, T, 12] (optional)
            actions: Action sequence [B, T_a, action_dim] (optional)
            
        Returns:
            Dict containing:
            - 'world_state': Current world state [B, state_dim]
            - 'fused_features': Fused multi-modal features [B, T, D]
            - 'predicted_states': Future states if actions provided
            - 'uncertainties': Prediction uncertainties
        """
        B = vision.shape[0]
        device = vision.device
        
        # Handle missing modalities with zeros
        if audio is None:
            audio = torch.zeros(B, self.config.latent_dim, device=device)
        if proprio is None:
            proprio = torch.zeros(B, 1, 12, device=device)
        
        # Build world state
        fused, attn = self.encode_multimodal(vision, audio, proprio)
        world_state, temporal_features = self.temporal_model(fused)
        
        results = {
            'world_state': world_state,
            'fused_features': fused,
            'temporal_features': temporal_features,
        }
        
        # Imagination if actions provided
        if actions is not None:
            predicted_states, uncertainties = self.imagine(world_state, actions)
            results['predicted_states'] = predicted_states
            results['uncertainties'] = uncertainties
        
        return results
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory system statistics."""
        return self.memory.get_stats()
    
    def save_memory(self, path: str) -> None:
        """Save memory to disk."""
        self.memory.save(path)
    
    def load_memory(self, path: str) -> None:
        """Load memory from disk."""
        self.memory.load(path)
