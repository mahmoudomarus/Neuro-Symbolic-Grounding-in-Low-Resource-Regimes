#!/usr/bin/env python3
"""
Quick verification script for the NSCA Cognitive Architecture.

Run this to verify all components are working:
    python verify_world_model.py

This tests:
1. Layer 0: World Model (perception, memory, dynamics)
2. Layer 1: Semantic Properties
3. Layer 2: Causal Reasoning
4. Layer 3: Motivation/Drives
5. Layer 4: Language Integration
6. Unified Cognitive Agent
"""
import sys
sys.path.insert(0, '.')

import torch

def test_imports():
    """Test all module imports."""
    print("Testing imports...")
    
    # Layer 0: World Model
    from src.priors.visual_prior import ColorOpponencyPrior, GaborPrior, DepthCuesPrior
    from src.priors.audio_prior import AuditoryPrior, OnsetDetector
    from src.priors.spatial_prior import SpatialPrior3D
    from src.priors.temporal_prior import TemporalPrior
    from src.encoders.vision_encoder import VisionEncoderWithPriors
    from src.encoders.audio_encoder import AudioEncoderWithPriors
    from src.encoders.proprio_encoder import ProprioEncoder
    from src.fusion.cross_modal import CrossModalFusion
    from src.memory.dual_memory import DualMemorySystem
    from src.learning.meta_learner import MetaLearner, PrototypicalNetworks
    from src.world_model.temporal_world_model import TemporalWorldModel
    from src.world_model.enhanced_dynamics import EnhancedDynamicsPredictor
    from src.world_model.unified_world_model import UnifiedWorldModel
    print("  ✓ Layer 0: World Model")
    
    # Layer 1: Semantic Properties
    from src.semantics.property_layer import PropertyLayer, PropertyConfig
    from src.semantics.affordances import AffordanceDetector
    from src.semantics.categories import CategoryClassifier
    print("  ✓ Layer 1: Semantic Properties")
    
    # Layer 2: Causal Reasoning
    from src.reasoning.causal_layer import CausalReasoner, CausalConfig
    from src.reasoning.intuitive_physics import IntuitivePhysics
    from src.reasoning.counterfactual import CounterfactualReasoner
    print("  ✓ Layer 2: Causal Reasoning")
    
    # Layer 3: Motivation
    from src.motivation.drive_system import DriveSystem, DriveConfig
    from src.motivation.intrinsic_reward import IntrinsicRewardComputer
    from src.motivation.attention import AttentionAllocator
    print("  ✓ Layer 3: Motivation/Drives")
    
    # Layer 4: Language
    from src.language.llm_integration import LanguageGrounding, LanguageConfig
    print("  ✓ Layer 4: Language Integration")
    
    # Unified Agent
    from src.cognitive_agent import CognitiveAgent, create_cognitive_agent
    print("  ✓ Unified Cognitive Agent")
    
    return True


def test_world_model():
    """Test Layer 0: World Model."""
    print("\nTesting Layer 0: World Model...")
    
    from src.world_model.unified_world_model import UnifiedWorldModel, WorldModelConfig
    from src.encoders.vision_encoder import VisionEncoderConfig
    from src.encoders.audio_encoder import AudioEncoderConfig
    from src.encoders.proprio_encoder import ProprioEncoderConfig
    from src.fusion.cross_modal import FusionConfig
    from src.world_model.temporal_world_model import TemporalWorldModelConfig
    from src.world_model.enhanced_dynamics import EnhancedDynamicsConfig
    
    LATENT_DIM = 128
    STATE_DIM = 64
    
    config = WorldModelConfig(latent_dim=LATENT_DIM, state_dim=STATE_DIM, action_dim=16)
    config.vision = VisionEncoderConfig(input_height=64, input_width=64, latent_dim=LATENT_DIM)
    config.audio = AudioEncoderConfig(sample_rate=16000, n_mels=40, latent_dim=64, output_dim=LATENT_DIM)
    config.proprio = ProprioEncoderConfig(input_dim=12, output_dim=LATENT_DIM)
    config.fusion = FusionConfig(dim=LATENT_DIM, num_heads=4, num_layers=2)
    config.temporal = TemporalWorldModelConfig(dim=LATENT_DIM, num_heads=4, num_layers=2, max_seq_len=16, state_dim=STATE_DIM)
    config.dynamics = EnhancedDynamicsConfig(state_dim=STATE_DIM, action_dim=16, hidden_dim=128)
    
    model = UnifiedWorldModel(config)
    
    batch_size = 2
    vision = torch.randn(batch_size, 2, 3, 64, 64)
    audio = torch.randn(batch_size, 16000)
    proprio = torch.randn(batch_size, 2, 12)
    actions = torch.randn(batch_size, 2, 16)
    
    results = model(vision, audio, proprio, actions)
    
    print(f"  ✓ World state: {results['world_state'].shape}")
    print(f"  ✓ Predictions: {results['predicted_states'].shape}")
    
    return model, config


def test_semantic_properties():
    """Test Layer 1: Semantic Properties."""
    print("\nTesting Layer 1: Semantic Properties...")
    
    from src.semantics.property_layer import PropertyLayer, PropertyConfig
    from src.semantics.affordances import AffordanceDetector, AffordanceConfig
    from src.semantics.categories import CategoryClassifier
    
    config = PropertyConfig(world_state_dim=64, audio_dim=64, proprio_dim=64)
    layer = PropertyLayer(config)
    
    world_state = torch.randn(2, 64)
    props, embed = layer(world_state)
    
    print(f"  ✓ Properties: hardness={props.hardness[0].item():.3f}, animacy={props.animacy[0].item():.3f}")
    
    # Affordances
    aff_detector = AffordanceDetector(AffordanceConfig())
    affordances = aff_detector(props)
    print(f"  ✓ Top affordances: {affordances.top_affordances(3)}")
    
    # Categories
    classifier = CategoryClassifier()
    _, scores = classifier(props)
    print(f"  ✓ Category: {scores.primary_category().value}")
    
    return True


def test_causal_reasoning():
    """Test Layer 2: Causal Reasoning."""
    print("\nTesting Layer 2: Causal Reasoning...")
    
    from src.reasoning.causal_layer import CausalReasoner, CausalConfig
    from src.reasoning.intuitive_physics import IntuitivePhysics
    from src.reasoning.counterfactual import CounterfactualReasoner
    
    # Causal reasoner
    config = CausalConfig(state_dim=64, action_dim=16)
    reasoner = CausalReasoner(config)
    
    state_before = torch.randn(2, 64)
    state_after = torch.randn(2, 64)
    action = torch.randn(2, 16)
    
    result = reasoner(state_before, state_after, action)
    print(f"  ✓ Intervention prob: {result['intervention_prob'].mean().item():.3f}")
    
    # Intuitive physics
    physics = IntuitivePhysics(64)
    is_supported, motion, *_ = physics.gravity(state_before)
    print(f"  ✓ Gravity prediction: supported={is_supported[0].item()}")
    
    # Counterfactual
    cf = CounterfactualReasoner(64)
    cf_state = cf.intervene(state_before, 0, 1.0)
    print(f"  ✓ Counterfactual state: {cf_state.shape}")
    
    return True


def test_motivation():
    """Test Layer 3: Motivation/Drives."""
    print("\nTesting Layer 3: Motivation/Drives...")
    
    from src.motivation.drive_system import DriveSystem, DriveConfig
    from src.motivation.intrinsic_reward import IntrinsicRewardComputer
    from src.motivation.attention import AttentionAllocator, DriveState
    
    # Drive system
    config = DriveConfig(state_dim=64)
    drives = DriveSystem(config)
    
    state = torch.randn(2, 64)
    drive_state, motivation = drives(state, torch.rand(2))
    
    print(f"  ✓ Curiosity: {drive_state.curiosity_level:.3f}")
    print(f"  ✓ Most urgent: {drive_state.most_urgent().value}")
    
    # Intrinsic reward
    reward_computer = IntrinsicRewardComputer(64)
    reward, components = reward_computer(state, torch.randn(2, 64), torch.randn(2, 32))
    print(f"  ✓ Intrinsic reward: {reward.mean().item():.3f}")
    
    # Attention
    attention = AttentionAllocator(64)
    attn, _ = attention(state, drive_state)
    print(f"  ✓ Attention: {attn.shape}")
    
    return True


def test_language():
    """Test Layer 4: Language Integration."""
    print("\nTesting Layer 4: Language Integration...")
    
    from src.language.llm_integration import LanguageGrounding, LanguageConfig
    
    config = LanguageConfig(use_external_llm=False)
    lang = LanguageGrounding(config)
    
    # Ground word
    rock_props = lang.ground_word('rock')
    print(f"  ✓ 'rock' grounded: hardness={rock_props[0].item():.2f}")
    
    # Describe concept
    props = torch.tensor([[0.9, 0.7, 0.3, 0.0, 0.9, 0.0, 0.7, 0.5, 0.0]])
    desc = lang.describe_concept(props)
    print(f"  ✓ Description: {desc[0]}")
    
    # Find matching word
    matched, conf = lang.find_matching_word(props)
    print(f"  ✓ Matched concept: {matched} (conf: {conf:.3f})")
    
    return True


def test_cognitive_agent():
    """Test the unified Cognitive Agent."""
    print("\nTesting Unified Cognitive Agent...")
    
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
    
    agent = create_cognitive_agent(world_config, use_llm=False)
    
    # Test perception
    vision = torch.randn(1, 2, 3, 32, 32)
    audio = torch.randn(1, 8000)
    proprio = torch.randn(1, 2, 12)
    
    result = agent.perceive(vision, audio, proprio)
    print(f"  ✓ Perception complete")
    print(f"    - World state: {result['world_state'].shape}")
    print(f"    - Properties: hardness={result['properties'].hardness.item():.2f}")
    print(f"    - Description: {result['description']}")
    print(f"    - Motivation: {result['motivation'].item():.2f}")
    
    # Test what_is_this
    what = agent.what_is_this()
    print(f"  ✓ What is this: {what['description']}")
    
    # Test imagination
    actions = torch.randn(1, 2, 8)
    imag = agent.imagine(actions)
    print(f"  ✓ Imagination: {imag['predicted_states'].shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in agent.parameters())
    print(f"  ✓ Total parameters: {total_params:,}")
    
    return True


def main():
    print("=" * 60)
    print("NSCA Cognitive Architecture Verification")
    print("=" * 60)
    
    try:
        test_imports()
        test_world_model()
        test_semantic_properties()
        test_causal_reasoning()
        test_motivation()
        test_language()
        test_cognitive_agent()
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
        print("\nThe cognitive architecture is ready!")
        print("\nArchitecture Summary:")
        print("  Layer 0: World Model - Multi-modal perception with innate priors")
        print("  Layer 1: Semantic Properties - Hardness, weight, animacy extraction")
        print("  Layer 2: Causal Reasoning - Why things happen, intuitive physics")
        print("  Layer 3: Motivation - Curiosity, competence, intrinsic rewards")
        print("  Layer 4: Language - Concept grounding, LLM integration")
        print("\nNext steps:")
        print("  1. Rent a GPU (A100 recommended)")
        print("  2. Run: python scripts/train_world_model.py --config configs/training_config.yaml")
        print("  3. Monitor with: wandb")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')
    exit(main())
