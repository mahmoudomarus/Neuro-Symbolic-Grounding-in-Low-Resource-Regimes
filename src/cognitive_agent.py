"""
Unified Cognitive Agent - Complete layered cognitive architecture.

Integrates all layers:
- Layer 0: World Model (perception, memory, dynamics)
- Layer 1: Semantic Properties (hardness, weight, animacy)
- Layer 2: Causal Reasoning (why things happen)
- Layer 3: Drive System (motivation, curiosity, competence)
- Layer 4: Language Integration (grounding with LLM)

This is the "brain" that perceives, understands, remembers,
reasons, and communicates.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Layer 0: World Model
from src.world_model.unified_world_model import UnifiedWorldModel, WorldModelConfig

# Layer 1: Semantic Properties
from src.semantics.property_layer import PropertyLayer, PropertyConfig, PropertyVector
from src.semantics.affordances import AffordanceDetector, AffordanceConfig
from src.semantics.categories import CategoryClassifier

# Layer 2: Causal Reasoning
from src.reasoning.causal_layer import CausalReasoner, CausalConfig
from src.reasoning.intuitive_physics import IntuitivePhysics
from src.reasoning.counterfactual import CounterfactualReasoner

# Layer 3: Drive System
from src.motivation.drive_system import DriveSystem, DriveConfig, DriveState
from src.motivation.intrinsic_reward import IntrinsicRewardComputer
from src.motivation.attention import AttentionAllocator

# Layer 4: Language Integration
from src.language.llm_integration import LanguageGrounding, LanguageConfig


@dataclass
class CognitiveConfig:
    """Configuration for the complete cognitive agent."""
    # Layer 0: World Model
    world_model: WorldModelConfig = field(default_factory=WorldModelConfig)
    
    # Layer 1: Semantic Properties
    property_layer: PropertyConfig = field(default_factory=lambda: PropertyConfig(
        world_state_dim=256,
        audio_dim=256,
        proprio_dim=256,
        hidden_dim=512,
    ))
    affordance: AffordanceConfig = field(default_factory=AffordanceConfig)
    
    # Layer 2: Causal Reasoning
    causal: CausalConfig = field(default_factory=lambda: CausalConfig(
        state_dim=256,
        action_dim=32,
        hidden_dim=512,
    ))
    
    # Layer 3: Drive System
    drive: DriveConfig = field(default_factory=lambda: DriveConfig(
        state_dim=256,
        hidden_dim=128,
    ))
    
    # Layer 4: Language
    language: LanguageConfig = field(default_factory=lambda: LanguageConfig(
        concept_dim=256,
        property_dim=9,
        hidden_dim=512,
        use_external_llm=False,
    ))


class CognitiveAgent(nn.Module):
    """
    Complete Cognitive Agent with layered architecture.
    
    Layer 0 (World Model): Perceives multi-modal input, builds world state
    Layer 1 (Properties): Extracts semantic properties from world state
    Layer 2 (Causal): Understands why things happen
    Layer 3 (Drives): Provides motivation for learning and action
    Layer 4 (Language): Grounds concepts in language, uses LLM for reasoning
    
    This is fundamentally different from current AI:
    - Not learning everything from scratch
    - Has innate priors (color, space, physics)
    - Understands properties, not just patterns
    - Learns causation, not just correlation
    - Has intrinsic motivation to learn
    """
    
    def __init__(self, config: CognitiveConfig) -> None:
        super().__init__()
        
        self.config = config
        
        # ===== LAYER 0: WORLD MODEL =====
        self.world_model = UnifiedWorldModel(config.world_model)
        
        # ===== LAYER 1: SEMANTIC PROPERTIES =====
        self.property_layer = PropertyLayer(config.property_layer)
        self.affordance_detector = AffordanceDetector(config.affordance)
        self.category_classifier = CategoryClassifier()
        
        # ===== LAYER 2: CAUSAL REASONING =====
        self.causal_reasoner = CausalReasoner(config.causal)
        self.intuitive_physics = IntuitivePhysics(config.causal.state_dim)
        self.counterfactual = CounterfactualReasoner(
            state_dim=config.causal.state_dim,
            hidden_dim=config.causal.hidden_dim,
        )
        
        # ===== LAYER 3: DRIVE SYSTEM =====
        self.drive_system = DriveSystem(config.drive)
        self.intrinsic_reward = IntrinsicRewardComputer(config.drive.state_dim)
        self.attention = AttentionAllocator(config.drive.state_dim)
        
        # ===== LAYER 4: LANGUAGE =====
        self.language = LanguageGrounding(config.language)
        
        # State tracking
        self.previous_state: Optional[torch.Tensor] = None
        self.current_state: Optional[torch.Tensor] = None
        self.current_properties: Optional[PropertyVector] = None
    
    def perceive(
        self,
        vision: torch.Tensor,
        audio: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Full perception pipeline through all layers.
        
        Args:
            vision: Visual input [B, T, C, H, W] or [B, C, H, W]
            audio: Audio input [B, T_samples] (optional)
            proprio: Proprioception [B, T, 12] or [B, 12] (optional)
            
        Returns:
            Complete perception result with all layer outputs
        """
        B = vision.shape[0]
        device = vision.device
        
        # Store previous state
        self.previous_state = self.current_state
        
        # ===== LAYER 0: Build world state =====
        world_result = self.world_model(vision, audio, proprio)
        world_state = world_result['world_state']
        self.current_state = world_state
        
        # Get individual modality features
        vision_features = self.world_model.encode_vision(
            vision if vision.dim() == 4 else vision[:, -1]
        )
        audio_features = None
        if audio is not None:
            audio_features = self.world_model.encode_audio(audio)
        proprio_features = None
        if proprio is not None:
            proprio_features = self.world_model.encode_proprio(
                proprio if proprio.dim() == 2 else proprio[:, -1]
            )
        
        # ===== LAYER 1: Extract properties =====
        properties, property_embed = self.property_layer(
            world_state,
            audio_features,
            proprio_features,
            self.previous_state,
        )
        self.current_properties = properties
        
        # Get affordances
        affordances = self.affordance_detector(properties)
        
        # Get category
        category_logits, category_scores = self.category_classifier(properties)
        
        # ===== LAYER 2: Causal analysis =====
        causal_result = None
        if self.previous_state is not None:
            causal_result = self.causal_reasoner(
                self.previous_state,
                world_state,
                None,  # No action in perception
            )
            
            # Check physics violations
            physics_violations = self.intuitive_physics.check_all(
                self.previous_state,
                world_state,
            )
        else:
            physics_violations = []
        
        # ===== LAYER 3: Drive update =====
        # Compute prediction error for curiosity
        if self.previous_state is not None:
            pred_error = (world_state - self.previous_state).norm(dim=-1)
        else:
            pred_error = torch.zeros(B, device=device)
        
        drive_state, motivation = self.drive_system(world_state, pred_error)
        
        # Compute attention allocation
        attention_weights, attention_components = self.attention(
            world_state.unsqueeze(1) if world_state.dim() == 2 else world_state,
            drive_state,
        )
        
        # ===== LAYER 4: Language description =====
        descriptions = self.language.describe_concept(properties.to_tensor())
        matched_concept, match_confidence = self.language.find_matching_word(
            properties.to_tensor()
        )
        
        return {
            # Layer 0
            'world_state': world_state,
            'fused_features': world_result.get('fused_features'),
            
            # Layer 1
            'properties': properties,
            'property_embedding': property_embed,
            'affordances': affordances,
            'category': category_scores,
            
            # Layer 2
            'causal_analysis': causal_result,
            'physics_violations': physics_violations,
            
            # Layer 3
            'drive_state': drive_state,
            'motivation': motivation,
            'attention': attention_weights,
            
            # Layer 4
            'description': descriptions[0] if descriptions else "",
            'matched_concept': matched_concept,
            'match_confidence': match_confidence,
        }
    
    def act(
        self,
        action: torch.Tensor,
        predicted_next_state: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Process an action and update causal understanding.
        
        Args:
            action: Action taken [B, action_dim]
            predicted_next_state: What we expected to happen
            
        Returns:
            Action processing results including causal learning
        """
        if self.current_state is None:
            raise ValueError("Must call perceive() before act()")
        
        # Learn causal relationship
        causal_relation = None
        if self.previous_state is not None:
            causal_relation = self.causal_reasoner.learn_from_intervention(
                self.previous_state,
                action,
                self.current_state,
            )
        
        # Compute intrinsic reward
        intrinsic_r = None
        if self.previous_state is not None:
            intrinsic_r, reward_components = self.intrinsic_reward(
                self.previous_state,
                self.current_state,
                action,
            )
        
        # Update competence if we had a prediction
        if predicted_next_state is not None:
            pred_accuracy = self.drive_system.competence.record_outcome(
                predicted_next_state,
                self.current_state,
            )
        else:
            pred_accuracy = None
        
        return {
            'causal_relation': causal_relation,
            'intrinsic_reward': intrinsic_r,
            'prediction_accuracy': pred_accuracy,
        }
    
    def imagine(
        self,
        actions: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Imagine future states given planned actions.
        
        Uses Layer 0 dynamics + Layer 2 causal reasoning.
        """
        if self.current_state is None:
            raise ValueError("Must call perceive() before imagine()")
        
        # Layer 0: Dynamics prediction
        predicted_states, uncertainties = self.world_model.imagine(
            self.current_state,
            actions,
        )
        
        # Layer 2: Check physics plausibility
        physics_ok = []
        for t in range(predicted_states.shape[1] - 1):
            violations = self.intuitive_physics.check_all(
                predicted_states[:, t],
                predicted_states[:, t + 1],
            )
            physics_ok.append(len(violations) == 0)
        
        return {
            'predicted_states': predicted_states,
            'uncertainties': uncertainties,
            'physics_plausible': physics_ok,
        }
    
    def why_did_this_happen(self) -> Dict[str, Any]:
        """
        Ask why the current state occurred.
        
        Uses Layer 2 causal reasoning.
        """
        if self.current_state is None or self.previous_state is None:
            return {'error': 'Need previous and current state'}
        
        causal_type, confidence, explanation = self.causal_reasoner.why_did_this_happen(
            self.previous_state,
            self.current_state,
        )
        
        return {
            'causal_type': causal_type.value,
            'confidence': confidence,
            'explanation': explanation,
        }
    
    def what_is_this(self) -> Dict[str, Any]:
        """
        Describe what we're perceiving.
        
        Uses Layer 1 properties + Layer 4 language.
        """
        if self.current_properties is None:
            return {'error': 'Need to perceive first'}
        
        # Get description
        descriptions = self.language.describe_concept(self.current_properties.to_tensor())
        
        # Get matched concept
        matched, confidence = self.language.find_matching_word(
            self.current_properties.to_tensor()
        )
        
        # Get affordances
        affordances = self.affordance_detector(self.current_properties)
        top_affordances = affordances.top_affordances(3)
        
        # Get category
        _, category = self.category_classifier(self.current_properties)
        
        return {
            'description': descriptions[0] if descriptions else "Unknown object",
            'matched_concept': matched,
            'confidence': confidence,
            'category': category.primary_category().value,
            'affordances': top_affordances,
            'properties': self.language.verbalizer.PROPERTY_ADJECTIVES,
        }
    
    def answer_question(self, question: str) -> str:
        """
        Answer a question about current perception.
        
        Uses all layers + LLM if available.
        """
        if self.current_properties is None:
            return "I haven't perceived anything yet."
        
        return self.language.answer_property_question(
            self.current_properties.to_tensor(),
            question,
        )
    
    def should_explore(self) -> bool:
        """Should the agent explore (based on drives)?"""
        return self.drive_system.should_explore()
    
    def get_motivation(self) -> DriveState:
        """Get current drive/motivation state."""
        return self.drive_system.drive_state
    
    def remember(self, label: Optional[str] = None) -> None:
        """Store current state in memory."""
        if self.current_state is None:
            return
        
        metadata = {'label': label} if label else {}
        
        if self.current_properties is not None:
            # Find matching concept
            matched, conf = self.language.find_matching_word(
                self.current_properties.to_tensor()
            )
            if conf > 0.5:
                metadata['matched_concept'] = matched
        
        self.world_model.remember(self.current_state, metadata)
    
    def recall(self, query: Optional[str] = None) -> Dict[str, Any]:
        """Recall from memory."""
        if query:
            # Ground query in properties
            query_props = self.language.grounder(query)
            # This is simplified - would need proper query embedding
            return self.world_model.recall(
                query_props.unsqueeze(0).expand(1, self.config.world_model.state_dim)[:, :self.config.world_model.state_dim]
            )
        elif self.current_state is not None:
            return self.world_model.recall(self.current_state)
        return {}
    
    def learn_new_concept(
        self,
        examples: torch.Tensor,
        concept_name: str,
    ) -> None:
        """
        Learn a new concept from few examples.
        
        Args:
            examples: Example images [K, C, H, W]
            concept_name: Name for the concept
        """
        self.world_model.learn_new_concept(examples, concept_name)
    
    def forward(
        self,
        vision: torch.Tensor,
        audio: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Full forward pass: perceive, optionally act.
        
        Returns comprehensive output from all layers.
        """
        # Perceive
        perception = self.perceive(vision, audio, proprio)
        
        # Act if action provided
        if action is not None:
            action_result = self.act(action)
            perception['action_result'] = action_result
        
        return perception


def create_cognitive_agent(
    world_model_config: Optional[WorldModelConfig] = None,
    use_llm: bool = False,
    llm_model: str = "gpt-3.5-turbo",
) -> CognitiveAgent:
    """
    Factory function to create a cognitive agent.
    
    Args:
        world_model_config: Custom world model config (or use defaults)
        use_llm: Whether to enable LLM integration
        llm_model: Which LLM to use
        
    Returns:
        Configured CognitiveAgent
    """
    config = CognitiveConfig()
    
    if world_model_config:
        config.world_model = world_model_config
    
    # Align dimensions across all layers
    state_dim = config.world_model.state_dim
    latent_dim = config.world_model.latent_dim
    action_dim = config.world_model.action_dim
    
    # Layer 1: Property layer needs to match world model dimensions
    config.property_layer.world_state_dim = state_dim
    config.property_layer.audio_dim = latent_dim  # Audio encoder outputs latent_dim
    config.property_layer.proprio_dim = latent_dim  # Proprio encoder outputs latent_dim
    
    # Layer 2: Causal reasoning
    config.causal.state_dim = state_dim
    config.causal.action_dim = action_dim
    
    # Layer 3: Drive system
    config.drive.state_dim = state_dim
    
    # Layer 4: Language
    config.language.concept_dim = state_dim
    config.language.use_external_llm = use_llm
    config.language.llm_model = llm_model
    
    return CognitiveAgent(config)
