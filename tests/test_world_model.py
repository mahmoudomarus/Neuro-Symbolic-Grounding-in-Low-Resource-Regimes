#!/usr/bin/env python3
"""
Unit tests for the Unified World Model components.

Tests all major components:
- Visual priors (color opponency, Gabor, depth)
- Audio priors (mel spectrogram, onset)
- Spatial and temporal priors
- Encoders (vision, audio, proprio)
- Cross-modal fusion
- Temporal world model
- Dynamics predictor
- Dual memory system
- Meta-learner

Run with: pytest tests/test_world_model.py -v
"""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


class TestVisualPriors:
    """Tests for visual innate priors."""
    
    def test_color_opponency_prior(self):
        """Test color opponency transformation."""
        from src.priors.visual_prior import ColorOpponencyPrior
        
        prior = ColorOpponencyPrior()
        
        # Test input
        rgb = torch.randn(2, 3, 64, 64)
        
        # Forward pass
        opponent = prior(rgb)
        
        # Check output shape
        assert opponent.shape == rgb.shape, f"Expected {rgb.shape}, got {opponent.shape}"
        
        # Check that transform is not learnable
        assert not any(p.requires_grad for p in prior.parameters())
    
    def test_gabor_prior(self):
        """Test Gabor filter bank."""
        from src.priors.visual_prior import GaborPrior
        
        prior = GaborPrior(num_orientations=8, num_scales=4, kernel_size=15)
        
        # Test input (single channel)
        x = torch.randn(2, 1, 64, 64)
        
        # Forward pass
        gabor_features = prior(x)
        
        # Check output shape
        expected_filters = 8 * 4  # orientations * scales
        assert gabor_features.shape == (2, expected_filters, 64, 64)
        
        # Check filters are fixed
        assert prior.filters.requires_grad == False
    
    def test_depth_cues_prior(self):
        """Test depth cues prior."""
        from src.priors.visual_prior import DepthCuesPrior
        
        prior = DepthCuesPrior(height=64, width=64)
        
        # Test input
        x = torch.randn(2, 3, 64, 64)
        
        # Forward pass
        depth = prior(x)
        
        # Check output shape
        assert depth.shape == (2, 1, 64, 64)
        
        # Check depth values are in [0, 1]
        assert depth.min() >= 0 and depth.max() <= 1


class TestAudioPriors:
    """Tests for audio innate priors."""
    
    def test_auditory_prior(self):
        """Test mel spectrogram transformation."""
        from src.priors.audio_prior import AuditoryPrior
        
        prior = AuditoryPrior(sample_rate=16000, n_mels=80, n_fft=400)
        
        # Test input (5 seconds of audio)
        waveform = torch.randn(2, 16000 * 5)
        
        # Forward pass
        mel = prior(waveform)
        
        # Check output shape
        assert mel.dim() == 3  # [B, n_mels, T']
        assert mel.shape[0] == 2
        assert mel.shape[1] == 80
    
    def test_onset_detector(self):
        """Test onset detection."""
        from src.priors.audio_prior import OnsetDetector
        
        detector = OnsetDetector(n_mels=80)
        
        # Test input (mel spectrogram)
        mel = torch.randn(2, 80, 100)
        
        # Forward pass
        onset = detector(mel)
        
        # Check output shape
        assert onset.shape == (2, 1, 100)


class TestSpatialPriors:
    """Tests for spatial priors."""
    
    def test_spatial_prior_3d(self):
        """Test 3D spatial prior."""
        from src.priors.spatial_prior import SpatialPrior3D
        
        prior = SpatialPrior3D(dim=64, max_height=16, max_width=16)
        
        # Test input
        features = torch.randn(2, 64, 16, 16)
        
        # Forward pass
        output = prior(features)
        
        # Check output shape preserved
        assert output.shape == features.shape


class TestTemporalPriors:
    """Tests for temporal priors."""
    
    def test_temporal_prior(self):
        """Test temporal prior with position encoding and causal mask."""
        from src.priors.temporal_prior import TemporalPrior
        
        prior = TemporalPrior(max_seq_len=64, dim=128)
        
        # Test input
        x = torch.randn(2, 32, 128)
        
        # Forward pass
        x_with_time, mask = prior(x)
        
        # Check output shape
        assert x_with_time.shape == x.shape
        
        # Check causal mask
        assert mask.shape == (32, 32)
        assert mask.dtype == torch.bool
        
        # Check upper triangle is True (masked)
        assert mask[0, 1] == True  # Can't attend to future
        assert mask[1, 0] == False  # Can attend to past


class TestEncoders:
    """Tests for multi-modal encoders."""
    
    def test_vision_encoder(self):
        """Test vision encoder with priors."""
        from src.encoders.vision_encoder import VisionEncoderWithPriors, VisionEncoderConfig
        
        config = VisionEncoderConfig(
            input_height=64,
            input_width=64,
            latent_dim=256,
        )
        encoder = VisionEncoderWithPriors(config)
        
        # Test input
        rgb = torch.randn(2, 3, 64, 64)
        
        # Forward pass
        features = encoder(rgb)
        
        # Check output shape
        assert features.dim() == 4  # [B, C, H', W']
        assert features.shape[0] == 2
        assert features.shape[1] == 256
    
    def test_audio_encoder(self):
        """Test audio encoder with priors."""
        from src.encoders.audio_encoder import AudioEncoderWithPriors, AudioEncoderConfig
        
        config = AudioEncoderConfig(
            sample_rate=16000,
            n_mels=80,
            latent_dim=128,
            output_dim=256,
        )
        encoder = AudioEncoderWithPriors(config)
        
        # Test input
        waveform = torch.randn(2, 16000 * 3)  # 3 seconds
        
        # Forward pass
        features = encoder(waveform)
        
        # Check output shape
        assert features.shape == (2, 256)
    
    def test_proprio_encoder(self):
        """Test proprioceptive encoder."""
        from src.encoders.proprio_encoder import ProprioEncoder, ProprioEncoderConfig
        
        config = ProprioEncoderConfig(
            input_dim=12,
            output_dim=256,
        )
        encoder = ProprioEncoder(config)
        
        # Test single timestep
        body_state = torch.randn(2, 12)
        features = encoder(body_state)
        assert features.shape == (2, 256)
        
        # Test sequence
        body_seq = torch.randn(2, 10, 12)
        features_seq = encoder(body_seq)
        assert features_seq.shape == (2, 10, 256)


class TestFusion:
    """Tests for cross-modal fusion."""
    
    def test_cross_modal_fusion(self):
        """Test cross-modal attention fusion."""
        from src.fusion.cross_modal import CrossModalFusion, FusionConfig
        
        config = FusionConfig(dim=256, num_heads=4, num_layers=2)
        fusion = CrossModalFusion(config)
        
        # Test inputs
        vision = torch.randn(2, 10, 256)
        audio = torch.randn(2, 5, 256)
        proprio = torch.randn(2, 10, 256)
        
        # Forward pass
        fused, attn = fusion(vision, audio, proprio, return_attention=True)
        
        # Check output shape
        total_tokens = 10 + 5 + 10
        assert fused.shape == (2, total_tokens, 256)
        
        # Check attention weights
        assert attn is not None
        assert len(attn) == 2  # num_layers


class TestTemporalWorldModel:
    """Tests for temporal world model."""
    
    def test_temporal_world_model(self):
        """Test temporal processing."""
        from src.world_model.temporal_world_model import TemporalWorldModel, TemporalWorldModelConfig
        
        config = TemporalWorldModelConfig(
            dim=256,
            num_heads=4,
            num_layers=2,
            max_seq_len=32,
            state_dim=128,
        )
        model = TemporalWorldModel(config)
        
        # Test input
        sequence = torch.randn(2, 16, 256)
        
        # Forward pass
        world_state, temporal_features = model(sequence)
        
        # Check output shapes
        assert world_state.shape == (2, 128)
        assert temporal_features.shape == (2, 16, 256)


class TestDynamicsPredictor:
    """Tests for dynamics predictor."""
    
    def test_enhanced_dynamics(self):
        """Test enhanced dynamics predictor."""
        from src.world_model.enhanced_dynamics import EnhancedDynamicsPredictor, EnhancedDynamicsConfig
        
        config = EnhancedDynamicsConfig(
            state_dim=128,
            action_dim=16,
            hidden_dim=256,
        )
        dynamics = EnhancedDynamicsPredictor(config)
        
        # Test single step
        state = torch.randn(2, 128)
        action = torch.randn(2, 16)
        
        next_state, uncertainty = dynamics(state, action)
        
        assert next_state.shape == (2, 128)
        assert uncertainty.shape == (2, 1)
    
    def test_trajectory_prediction(self):
        """Test multi-step trajectory prediction."""
        from src.world_model.enhanced_dynamics import EnhancedDynamicsPredictor, EnhancedDynamicsConfig
        
        config = EnhancedDynamicsConfig(state_dim=128, action_dim=16)
        dynamics = EnhancedDynamicsPredictor(config)
        
        # Test trajectory
        initial_state = torch.randn(2, 128)
        actions = torch.randn(2, 5, 16)  # 5 steps
        
        states, uncertainties = dynamics.predict_trajectory(initial_state, actions)
        
        assert states.shape == (2, 6, 128)  # T+1 states
        assert uncertainties.shape == (2, 5, 1)  # T uncertainties


class TestDualMemory:
    """Tests for dual memory system."""
    
    def test_episodic_storage(self):
        """Test episodic memory storage and recall."""
        from src.memory.dual_memory import DualMemorySystem
        
        memory = DualMemorySystem(dim=128, max_episodic=100)
        
        # Store some experiences
        for i in range(10):
            vector = torch.randn(128)
            metadata = {'label': f'class_{i % 3}', 'step': i}
            memory.store(vector, metadata)
        
        # Recall
        query = torch.randn(128)
        results = memory.recall_episodic(query, k=5)
        
        assert len(results) <= 5
        assert all(isinstance(r, tuple) for r in results)
    
    def test_semantic_consolidation(self):
        """Test semantic memory consolidation."""
        from src.memory.dual_memory import DualMemorySystem
        
        memory = DualMemorySystem(dim=128, consolidation_threshold=3)
        
        # Store multiple similar experiences with same label
        base_vector = torch.randn(128)
        for i in range(5):
            vector = base_vector + torch.randn(128) * 0.1
            memory.store(vector, {'label': 'test_concept'})
        
        # Check semantic memory was created
        assert 'test_concept' in memory.semantic
    
    def test_learn_new_concept(self):
        """Test direct concept learning."""
        from src.memory.dual_memory import DualMemorySystem
        
        memory = DualMemorySystem(dim=128)
        
        # Learn concept from examples
        examples = torch.randn(5, 128)
        memory.learn_new_concept(examples, 'new_concept')
        
        # Recall should find it
        query = examples.mean(dim=0)
        results = memory.recall_semantic(query, k=1)
        
        assert len(results) == 1
        assert results[0][0] == 'new_concept'


class TestMetaLearner:
    """Tests for meta-learning."""
    
    @pytest.mark.skip(reason="MAML gradient computation requires specific model architecture")
    def test_maml_adapt(self):
        """Test MAML-style adaptation."""
        from src.learning.meta_learner import MetaLearner
        
        # Simple test model
        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )
        
        meta_learner = MetaLearner(model, inner_lr=0.01, inner_steps=3)
        
        # Support set
        support_data = torch.randn(10, 64)
        support_labels = torch.randint(0, 2, (10,))
        
        # Adapt
        adapted_params = meta_learner.adapt(support_data, support_labels)
        
        # Check we got parameters
        assert len(adapted_params) > 0
    
    def test_prototypical_networks(self):
        """Test prototypical networks."""
        from src.learning.meta_learner import PrototypicalNetworks
        
        # Simple encoder
        encoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )
        
        proto_net = PrototypicalNetworks(encoder)
        
        # Support set
        support_data = torch.randn(10, 64)  # 2 classes, 5 each
        support_labels = torch.tensor([0]*5 + [1]*5)
        
        # Query set
        query_data = torch.randn(6, 64)
        
        # Classify
        predictions = proto_net.few_shot_classify(support_data, support_labels, query_data)
        
        assert predictions.shape == (6,)
        assert all(p in [0, 1] for p in predictions)


class TestUnifiedWorldModel:
    """Tests for the complete unified world model."""
    
    @pytest.fixture
    def aligned_config(self):
        """Create properly aligned configuration."""
        from src.world_model.unified_world_model import WorldModelConfig
        from src.encoders.vision_encoder import VisionEncoderConfig
        from src.encoders.audio_encoder import AudioEncoderConfig
        from src.encoders.proprio_encoder import ProprioEncoderConfig
        from src.fusion.cross_modal import FusionConfig
        from src.world_model.temporal_world_model import TemporalWorldModelConfig
        from src.world_model.enhanced_dynamics import EnhancedDynamicsConfig
        
        LATENT_DIM = 128
        STATE_DIM = 64
        ACTION_DIM = 16
        
        config = WorldModelConfig(latent_dim=LATENT_DIM, state_dim=STATE_DIM, action_dim=ACTION_DIM)
        config.vision = VisionEncoderConfig(input_height=64, input_width=64, latent_dim=LATENT_DIM)
        config.audio = AudioEncoderConfig(sample_rate=16000, n_mels=40, latent_dim=64, output_dim=LATENT_DIM)
        config.proprio = ProprioEncoderConfig(input_dim=12, output_dim=LATENT_DIM)
        config.fusion = FusionConfig(dim=LATENT_DIM, num_heads=4, num_layers=2)
        config.temporal = TemporalWorldModelConfig(dim=LATENT_DIM, num_heads=4, num_layers=2, max_seq_len=16, state_dim=STATE_DIM)
        config.dynamics = EnhancedDynamicsConfig(state_dim=STATE_DIM, action_dim=ACTION_DIM, hidden_dim=128)
        
        return config
    
    def test_model_creation(self, aligned_config):
        """Test model can be created."""
        from src.world_model.unified_world_model import UnifiedWorldModel
        
        model = UnifiedWorldModel(aligned_config)
        
        # Check all components exist
        assert hasattr(model, 'vision_encoder')
        assert hasattr(model, 'audio_encoder')
        assert hasattr(model, 'proprio_encoder')
        assert hasattr(model, 'fusion')
        assert hasattr(model, 'temporal_model')
        assert hasattr(model, 'dynamics')
        assert hasattr(model, 'memory')
    
    def test_forward_pass(self, aligned_config):
        """Test forward pass through complete model."""
        from src.world_model.unified_world_model import UnifiedWorldModel
        
        model = UnifiedWorldModel(aligned_config)
        
        # Test inputs
        vision = torch.randn(2, 4, 3, 64, 64)  # Video: B, T, C, H, W
        audio = torch.randn(2, 16000 * 2)  # 2 seconds
        proprio = torch.randn(2, 4, 12)  # B, T, 12
        actions = torch.randn(2, 3, aligned_config.action_dim)  # B, T_actions, action_dim
        
        # Forward
        results = model(vision, audio, proprio, actions)
        
        # Check outputs
        assert 'world_state' in results
        assert results['world_state'].shape == (2, aligned_config.state_dim)
        
        assert 'predicted_states' in results
        assert results['predicted_states'].shape == (2, 4, aligned_config.state_dim)  # T+1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
