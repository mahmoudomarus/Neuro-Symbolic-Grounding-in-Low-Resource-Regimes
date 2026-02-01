#!/usr/bin/env python3
"""
Quick verification script for the NSCA World Model.

Run this to verify all components are working:
    python verify_world_model.py

This tests:
1. All module imports
2. Model instantiation
3. Forward pass
4. Memory operations
5. Few-shot learning
"""
import sys
sys.path.insert(0, '.')

import torch

def test_imports():
    """Test all module imports."""
    print("Testing imports...")
    
    # Priors
    from src.priors.visual_prior import ColorOpponencyPrior, GaborPrior, DepthCuesPrior
    from src.priors.audio_prior import AuditoryPrior, OnsetDetector
    from src.priors.spatial_prior import SpatialPrior3D
    from src.priors.temporal_prior import TemporalPrior
    print("  ✓ Priors")
    
    # Encoders
    from src.encoders.vision_encoder import VisionEncoderWithPriors
    from src.encoders.audio_encoder import AudioEncoderWithPriors
    from src.encoders.proprio_encoder import ProprioEncoder
    print("  ✓ Encoders")
    
    # Fusion
    from src.fusion.cross_modal import CrossModalFusion
    print("  ✓ Fusion")
    
    # Memory
    from src.memory.dual_memory import DualMemorySystem
    print("  ✓ Memory")
    
    # Learning
    from src.learning.meta_learner import MetaLearner, PrototypicalNetworks
    print("  ✓ Learning")
    
    # World model
    from src.world_model.temporal_world_model import TemporalWorldModel
    from src.world_model.enhanced_dynamics import EnhancedDynamicsPredictor
    from src.world_model.unified_world_model import UnifiedWorldModel
    print("  ✓ World model")
    
    return True


def test_model_forward():
    """Test model forward pass."""
    print("\nTesting model forward pass...")
    
    from src.world_model.unified_world_model import UnifiedWorldModel, WorldModelConfig
    from src.encoders.vision_encoder import VisionEncoderConfig
    from src.encoders.audio_encoder import AudioEncoderConfig
    from src.encoders.proprio_encoder import ProprioEncoderConfig
    from src.fusion.cross_modal import FusionConfig
    from src.world_model.temporal_world_model import TemporalWorldModelConfig
    from src.world_model.enhanced_dynamics import EnhancedDynamicsConfig
    
    # Config with aligned dimensions
    LATENT_DIM = 256
    STATE_DIM = 128
    
    config = WorldModelConfig(latent_dim=LATENT_DIM, state_dim=STATE_DIM, action_dim=16)
    config.vision = VisionEncoderConfig(input_height=64, input_width=64, latent_dim=LATENT_DIM)
    config.audio = AudioEncoderConfig(sample_rate=16000, n_mels=40, latent_dim=128, output_dim=LATENT_DIM)
    config.proprio = ProprioEncoderConfig(input_dim=12, output_dim=LATENT_DIM)
    config.fusion = FusionConfig(dim=LATENT_DIM, num_heads=4, num_layers=2)
    config.temporal = TemporalWorldModelConfig(dim=LATENT_DIM, num_heads=4, num_layers=2, max_seq_len=32, state_dim=STATE_DIM)
    config.dynamics = EnhancedDynamicsConfig(state_dim=STATE_DIM, action_dim=16, hidden_dim=256)
    
    model = UnifiedWorldModel(config)
    
    # Forward pass
    batch_size = 2
    vision = torch.randn(batch_size, 4, 3, 64, 64)
    audio = torch.randn(batch_size, 16000 * 2)
    proprio = torch.randn(batch_size, 4, 12)
    actions = torch.randn(batch_size, 3, 16)
    
    results = model(vision, audio, proprio, actions)
    
    assert results['world_state'].shape == (batch_size, STATE_DIM)
    assert results['predicted_states'].shape == (batch_size, 4, STATE_DIM)
    
    print(f"  ✓ World state: {results['world_state'].shape}")
    print(f"  ✓ Predictions: {results['predicted_states'].shape}")
    
    return model


def test_memory(model):
    """Test memory operations."""
    print("\nTesting memory...")
    
    # Store some experiences
    for i in range(10):
        state = torch.randn(model.config.state_dim)
        model.remember(state, {'label': f'concept_{i % 3}', 'step': i})
    
    # Recall
    query = torch.randn(model.config.state_dim)
    results = model.recall(query)
    
    print(f"  ✓ Episodic memories: {len(results['episodic'])}")
    print(f"  ✓ Semantic concepts: {len(results['semantic'])}")
    
    # Learn new concept
    examples = torch.randn(5, model.config.state_dim)
    model.memory.learn_new_concept(examples, 'new_concept')
    
    stats = model.get_memory_stats()
    print(f"  ✓ Total semantic concepts: {stats['semantic_count']}")
    
    return True


def test_priors():
    """Test innate priors."""
    print("\nTesting innate priors...")
    
    from src.priors.visual_prior import ColorOpponencyPrior, GaborPrior
    from src.priors.audio_prior import AuditoryPrior
    
    # Color
    color_prior = ColorOpponencyPrior()
    rgb = torch.randn(1, 3, 64, 64)
    opponent = color_prior(rgb)
    assert opponent.shape == rgb.shape
    print("  ✓ Color opponency")
    
    # Gabor
    gabor_prior = GaborPrior()
    luminance = torch.randn(1, 1, 64, 64)
    edges = gabor_prior(luminance)
    assert edges.shape[1] == 32  # 8 orientations * 4 scales
    print("  ✓ Gabor filters")
    
    # Audio
    audio_prior = AuditoryPrior()
    waveform = torch.randn(1, 16000)
    mel = audio_prior(waveform)
    assert mel.dim() == 3
    print("  ✓ Mel spectrogram")
    
    return True


def main():
    print("=" * 50)
    print("NSCA World Model Verification")
    print("=" * 50)
    
    try:
        test_imports()
        model = test_model_forward()
        test_memory(model)
        test_priors()
        
        print("\n" + "=" * 50)
        print("✓ ALL TESTS PASSED")
        print("=" * 50)
        print("\nThe world model is ready for training!")
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
    exit(main())
