"""
Tests for the cognitive architecture layers.

Layer 1: Semantic Properties
Layer 2: Causal Reasoning  
Layer 3: Motivation/Drives
Layer 4: Language Integration
Unified: Cognitive Agent
"""

import pytest
import torch
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


# =============================================================================
# LAYER 1: SEMANTIC PROPERTIES
# =============================================================================

class TestPropertyLayer:
    """Test semantic property extraction."""
    
    def test_property_layer_creation(self):
        """Property layer can be created."""
        from src.semantics.property_layer import PropertyLayer, PropertyConfig
        
        config = PropertyConfig(
            world_state_dim=64,
            audio_dim=64,
            proprio_dim=64,
            hidden_dim=128,
        )
        layer = PropertyLayer(config)
        
        assert layer is not None
        assert hasattr(layer, 'hardness_extractor')
        assert hasattr(layer, 'weight_extractor')
        assert hasattr(layer, 'animacy_detector')
    
    def test_property_extraction(self):
        """Properties can be extracted from world state."""
        from src.semantics.property_layer import PropertyLayer, PropertyConfig
        
        config = PropertyConfig(world_state_dim=64, audio_dim=64, proprio_dim=64)
        layer = PropertyLayer(config)
        
        world_state = torch.randn(2, 64)
        audio = torch.randn(2, 64)
        proprio = torch.randn(2, 64)
        
        props, embed = layer(world_state, audio, proprio)
        
        # Check all properties exist and are valid
        assert props.hardness.shape == (2,)
        assert props.weight.shape == (2,)
        assert props.animacy.shape == (2,)
        assert (props.hardness >= 0).all() and (props.hardness <= 1).all()
        assert embed.shape == (2, 512)  # hidden_dim
    
    def test_property_vector_conversion(self):
        """Property vector can be converted to/from tensor."""
        from src.semantics.property_layer import PropertyVector
        
        props = PropertyVector(
            hardness=torch.tensor([0.8]),
            weight=torch.tensor([0.5]),
            size=torch.tensor([0.3]),
            animacy=torch.tensor([0.0]),
            rigidity=torch.tensor([0.9]),
            transparency=torch.tensor([0.1]),
            roughness=torch.tensor([0.6]),
            temperature=torch.tensor([0.5]),
            containment=torch.tensor([0.0]),
        )
        
        tensor = props.to_tensor()
        assert tensor.shape == (1, 9)
        
        props2 = PropertyVector.from_tensor(tensor)
        assert torch.allclose(props2.hardness, props.hardness)


class TestAffordances:
    """Test affordance detection."""
    
    def test_affordance_detector_creation(self):
        from src.semantics.affordances import AffordanceDetector, AffordanceConfig
        
        config = AffordanceConfig()
        detector = AffordanceDetector(config)
        assert detector is not None
    
    def test_affordance_detection(self):
        from src.semantics.affordances import AffordanceDetector, AffordanceConfig
        from src.semantics.property_layer import PropertyVector
        
        config = AffordanceConfig()
        detector = AffordanceDetector(config)
        
        # Small, light object should be graspable
        props = PropertyVector(
            hardness=torch.tensor([0.5]),
            weight=torch.tensor([0.2]),  # Light
            size=torch.tensor([0.2]),    # Small
            animacy=torch.tensor([0.0]),
            rigidity=torch.tensor([0.5]),
            transparency=torch.tensor([0.0]),
            roughness=torch.tensor([0.5]),
            temperature=torch.tensor([0.5]),
            containment=torch.tensor([0.0]),
        )
        
        affordances = detector(props)
        top = affordances.top_affordances(3)
        
        assert len(top) == 3
        assert all(isinstance(a[0], str) and isinstance(a[1], float) for a in top)


class TestCategories:
    """Test category classification."""
    
    def test_category_classifier(self):
        from src.semantics.categories import CategoryClassifier, FundamentalCategory
        from src.semantics.property_layer import PropertyVector
        
        classifier = CategoryClassifier()
        
        # Animate object (high animacy)
        animate_props = PropertyVector(
            hardness=torch.tensor([0.5]),
            weight=torch.tensor([0.5]),
            size=torch.tensor([0.5]),
            animacy=torch.tensor([0.9]),  # High animacy
            rigidity=torch.tensor([0.3]),
            transparency=torch.tensor([0.0]),
            roughness=torch.tensor([0.5]),
            temperature=torch.tensor([0.6]),
            containment=torch.tensor([0.0]),
        )
        
        logits, scores = classifier(animate_props)
        
        # Should classify as agent due to high animacy
        assert scores.primary_category() == FundamentalCategory.AGENT


# =============================================================================
# LAYER 2: CAUSAL REASONING
# =============================================================================

class TestCausalReasoning:
    """Test causal reasoning layer."""
    
    def test_causal_reasoner_creation(self):
        from src.reasoning.causal_layer import CausalReasoner, CausalConfig
        
        config = CausalConfig(state_dim=64, action_dim=16)
        reasoner = CausalReasoner(config)
        
        assert reasoner is not None
    
    def test_causal_analysis(self):
        from src.reasoning.causal_layer import CausalReasoner, CausalConfig
        
        config = CausalConfig(state_dim=64, action_dim=16)
        reasoner = CausalReasoner(config)
        
        state_before = torch.randn(2, 64)
        state_after = torch.randn(2, 64)
        action = torch.randn(2, 16)
        
        result = reasoner(state_before, state_after, action)
        
        assert 'intervention_prob' in result
        assert 'causal_strength' in result
        assert result['intervention_prob'].shape == (2,)
    
    def test_intervention_learning(self):
        from src.reasoning.causal_layer import CausalReasoner, CausalConfig, CausalType
        
        config = CausalConfig(state_dim=64, action_dim=16)
        reasoner = CausalReasoner(config)
        
        state_before = torch.randn(2, 64)
        action = torch.randn(2, 16)
        state_after = state_before + 0.1 * action.sum(dim=-1, keepdim=True)
        
        relation = reasoner.learn_from_intervention(state_before, action, state_after)
        
        assert relation.causal_type == CausalType.INTERVENTION


class TestIntuitivePhysics:
    """Test intuitive physics module."""
    
    def test_physics_creation(self):
        from src.reasoning.intuitive_physics import IntuitivePhysics
        
        physics = IntuitivePhysics(feature_dim=64)
        assert physics is not None
    
    def test_gravity_prior(self):
        from src.reasoning.intuitive_physics import IntuitivePhysics
        
        physics = IntuitivePhysics(feature_dim=64)
        
        object_state = torch.randn(2, 64)
        
        # New adaptive physics returns 3 values: is_supported, motion, diagnostics
        if physics.use_adaptive_priors:
            is_supported, expected_motion, diagnostics = physics.gravity(object_state)
            # Check critical period floor is respected
            assert diagnostics['prior_weight'].item() >= 0.3
        else:
            is_supported, expected_motion = physics.gravity(object_state)
        
        assert is_supported.shape == (2,)
        assert expected_motion.shape == (2, 3)


class TestCounterfactual:
    """Test counterfactual reasoning."""
    
    def test_counterfactual_reasoner(self):
        from src.reasoning.counterfactual import CounterfactualReasoner
        
        reasoner = CounterfactualReasoner(state_dim=64, hidden_dim=128)
        
        state = torch.randn(2, 64)
        factors = reasoner.encode_to_factors(state)
        
        assert factors.shape == (2, 16)  # num_factors
    
    def test_intervention(self):
        from src.reasoning.counterfactual import CounterfactualReasoner
        
        reasoner = CounterfactualReasoner(state_dim=64)
        
        state = torch.randn(2, 64)
        counterfactual_state = reasoner.intervene(state, factor_idx=0, new_value=1.0)
        
        assert counterfactual_state.shape == state.shape


# =============================================================================
# LAYER 3: MOTIVATION/DRIVES
# =============================================================================

class TestDriveSystem:
    """Test drive/motivation system."""
    
    def test_drive_system_creation(self):
        from src.motivation.drive_system import DriveSystem, DriveConfig
        
        config = DriveConfig(state_dim=64)
        system = DriveSystem(config)
        
        assert system is not None
    
    def test_drive_computation(self):
        from src.motivation.drive_system import DriveSystem, DriveConfig
        
        config = DriveConfig(state_dim=64)
        system = DriveSystem(config)
        
        state = torch.randn(2, 64)
        pred_error = torch.rand(2)
        
        drive_state, motivation = system(state, pred_error)
        
        assert 0 <= drive_state.curiosity_level <= 1
        assert 0 <= drive_state.energy_level <= 1
        assert motivation.shape == (2,)
    
    def test_curiosity_drive(self):
        from src.motivation.drive_system import CuriosityDrive
        
        curiosity = CuriosityDrive(state_dim=64)
        
        # Novel state should have higher curiosity
        state = torch.randn(2, 64)
        
        # Update memory with some states
        for _ in range(20):
            curiosity.update_memory(torch.randn(1, 64))
        
        novelty = curiosity.compute_novelty(state)
        
        assert novelty.shape == (2,)
        assert (novelty >= 0).all() and (novelty <= 1).all()


class TestIntrinsicReward:
    """Test intrinsic reward computation."""
    
    def test_intrinsic_reward(self):
        from src.motivation.intrinsic_reward import IntrinsicRewardComputer
        
        computer = IntrinsicRewardComputer(state_dim=64)
        
        state = torch.randn(2, 64)
        next_state = torch.randn(2, 64)
        action = torch.randn(2, 32)
        
        reward, components = computer(state, next_state, action)
        
        assert reward.shape == (2,)
        assert 'curiosity' in components.to_dict()
        assert 'competence' in components.to_dict()


class TestAttention:
    """Test attention allocation."""
    
    def test_attention_allocator(self):
        from src.motivation.attention import AttentionAllocator
        from src.motivation.drive_system import DriveState
        
        allocator = AttentionAllocator(feature_dim=64)
        drive_state = DriveState()
        
        features = torch.randn(2, 64)
        
        attention, components = allocator(features, drive_state)
        
        assert attention.shape == (2,)
        assert 'salience' in components
        assert 'novelty' in components


# =============================================================================
# LAYER 4: LANGUAGE INTEGRATION
# =============================================================================

class TestLanguageGrounding:
    """Test language-concept grounding."""
    
    def test_concept_verbalizer(self):
        from src.language.llm_integration import ConceptVerbalizer
        
        verbalizer = ConceptVerbalizer()
        
        props = torch.tensor([[0.9, 0.7, 0.3, 0.0, 0.9, 0.0, 0.7, 0.5, 0.0]])
        
        descriptions = verbalizer(props)
        
        assert len(descriptions) == 1
        assert 'hard' in descriptions[0].lower()
    
    def test_text_grounder(self):
        from src.language.llm_integration import LearnedGrounding
        
        grounder = LearnedGrounding()
        
        # New: LearnedGrounding starts EMPTY (no hard-coded concepts)
        # This is a key architectural change from the peer review
        assert len(grounder.get_grounded_concepts()) == 0
        
        # Unknown concept should return neutral values (learned predictor)
        unknown_props = grounder('qwertyuiop')
        assert unknown_props.shape == (9,)
        
        # Test learning from interaction
        grounder.learn_from_interaction(
            object_id='rock_001',
            action='strike',
            sensory_feedback={'audio_frequency': 0.9}
        )
        
        # Now 'rock_001' should be grounded
        assert 'rock_001' in grounder.get_grounded_concepts()
        rock_props = grounder('rock_001')
        assert rock_props[0] > 0.5  # Hardness should be elevated after interaction
    
    def test_language_grounding(self):
        from src.language.llm_integration import LanguageGrounding, LanguageConfig
        
        config = LanguageConfig(use_external_llm=False)
        grounding = LanguageGrounding(config)
        
        # Test concept matching
        rock_expected = grounding.ground_word('rock')
        matches, score = grounding.concept_matches_word(rock_expected, 'rock')
        
        assert score > 0.3  # Should match reasonably well (untrained model)


# =============================================================================
# UNIFIED COGNITIVE AGENT
# =============================================================================

class TestCognitiveAgent:
    """Test the complete cognitive agent."""
    
    @pytest.fixture
    def agent(self):
        from src.cognitive_agent import create_cognitive_agent
        from src.world_model.unified_world_model import WorldModelConfig
        from src.encoders.vision_encoder import VisionEncoderConfig
        from src.encoders.audio_encoder import AudioEncoderConfig
        from src.encoders.proprio_encoder import ProprioEncoderConfig
        from src.fusion.cross_modal import FusionConfig
        from src.world_model.temporal_world_model import TemporalWorldModelConfig
        from src.world_model.enhanced_dynamics import EnhancedDynamicsConfig
        
        LATENT_DIM = 64
        STATE_DIM = 32
        
        world_config = WorldModelConfig(latent_dim=LATENT_DIM, state_dim=STATE_DIM, action_dim=8)
        world_config.vision = VisionEncoderConfig(input_height=32, input_width=32, latent_dim=LATENT_DIM)
        world_config.audio = AudioEncoderConfig(sample_rate=8000, n_mels=20, latent_dim=32, output_dim=LATENT_DIM)
        world_config.proprio = ProprioEncoderConfig(input_dim=12, output_dim=LATENT_DIM)
        world_config.fusion = FusionConfig(dim=LATENT_DIM, num_heads=2, num_layers=1)
        world_config.temporal = TemporalWorldModelConfig(dim=LATENT_DIM, num_heads=2, num_layers=1, max_seq_len=8, state_dim=STATE_DIM)
        world_config.dynamics = EnhancedDynamicsConfig(state_dim=STATE_DIM, action_dim=8, hidden_dim=64)
        
        return create_cognitive_agent(world_config, use_llm=False)
    
    def test_agent_creation(self, agent):
        assert agent is not None
        assert hasattr(agent, 'world_model')
        assert hasattr(agent, 'property_layer')
        assert hasattr(agent, 'causal_reasoner')
        assert hasattr(agent, 'drive_system')
        assert hasattr(agent, 'language')
    
    def test_perception(self, agent):
        vision = torch.randn(2, 2, 3, 32, 32)
        audio = torch.randn(2, 8000)
        proprio = torch.randn(2, 2, 12)
        
        result = agent.perceive(vision, audio, proprio)
        
        assert 'world_state' in result
        assert 'properties' in result
        assert 'drive_state' in result
        assert 'description' in result
    
    def test_imagination(self, agent):
        vision = torch.randn(2, 2, 3, 32, 32)
        agent.perceive(vision)
        
        actions = torch.randn(2, 3, 8)
        result = agent.imagine(actions)
        
        assert 'predicted_states' in result
        assert result['predicted_states'].shape == (2, 4, 32)
    
    def test_what_is_this(self, agent):
        vision = torch.randn(1, 2, 3, 32, 32)
        agent.perceive(vision)
        
        what = agent.what_is_this()
        
        assert 'description' in what
        assert 'category' in what
        assert 'affordances' in what


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
