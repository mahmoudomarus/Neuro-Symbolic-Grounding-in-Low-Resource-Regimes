"""
Tests for the enhanced cross-modal fusion module.

Tests cover:
- Hierarchical fusion (early/mid/late)
- Contrastive alignment (CLIP-style)
- Temporal synchronization
- Cross-modal prediction
- EnhancedCrossModalFusion integration
"""

import pytest
import torch
import torch.nn as nn

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.fusion.cross_modal import (
    FusionConfig,
    ModalityEmbedding,
    CrossModalAttentionLayer,
    CrossModalFusion,
    ContrastiveAlignment,
    TemporalSynchronizer,
    CrossModalPredictor,
    EarlyFusion,
    MidFusion,
    LateFusion,
    HierarchicalFusion,
    EnhancedCrossModalFusion,
    create_fusion_module,
)


class TestContrastiveAlignment:
    """Test CLIP-style contrastive alignment."""
    
    def test_contrastive_forward(self):
        """Test contrastive alignment forward pass."""
        align = ContrastiveAlignment(dim=512, projection_dim=256, temperature=0.07)
        
        vision = torch.randn(4, 512)
        audio = torch.randn(4, 512)
        proprio = torch.randn(4, 512)
        
        result = align(vision, audio, proprio)
        
        assert 'vision_emb' in result
        assert 'audio_emb' in result
        assert 'proprio_emb' in result
        assert result['vision_emb'].shape == (4, 256)
        assert 'loss_va' in result
        assert 'loss_total' in result
    
    def test_contrastive_temporal_pooling(self):
        """Test contrastive alignment with temporal inputs."""
        align = ContrastiveAlignment(dim=512)
        
        # Temporal inputs
        vision = torch.randn(4, 8, 512)  # B, T, D
        audio = torch.randn(4, 16, 512)
        proprio = torch.randn(4, 8, 512)
        
        result = align(vision, audio, proprio)
        
        # Should pool to [B, D]
        assert result['vision_emb'].shape == (4, 256)
    
    def test_contrastive_loss_values(self):
        """Test that contrastive loss is reasonable."""
        align = ContrastiveAlignment(dim=512)
        
        vision = torch.randn(8, 512)
        audio = torch.randn(8, 512)
        proprio = torch.randn(8, 512)
        
        result = align(vision, audio, proprio)
        
        # Loss should be positive and finite
        assert result['loss_total'].item() > 0
        assert torch.isfinite(result['loss_total'])
    
    def test_similarity_computation(self):
        """Test similarity matrix computation."""
        align = ContrastiveAlignment(dim=512)
        
        emb1 = torch.randn(4, 256)
        emb2 = torch.randn(6, 256)
        
        sim = align.get_similarity(emb1, emb2)
        
        assert sim.shape == (4, 6)
        # Similarity should be between -1 and 1
        assert sim.min() >= -1.01
        assert sim.max() <= 1.01


class TestTemporalSynchronizer:
    """Test temporal synchronization across modalities."""
    
    def test_synchronize_basic(self):
        """Test basic temporal synchronization."""
        sync = TemporalSynchronizer(dim=512, max_length=100)
        
        features = torch.randn(4, 16, 512)
        target_length = 24
        
        output = sync(features, target_length)
        
        assert output.shape == (4, 24, 512)
    
    def test_synchronize_with_reference(self):
        """Test synchronization with reference features."""
        sync = TemporalSynchronizer(dim=512)
        
        features = torch.randn(4, 8, 512)
        reference = torch.randn(4, 16, 512)
        
        output = sync(features, target_length=16, reference=reference)
        
        assert output.shape == (4, 16, 512)
    
    def test_synchronize_modalities(self):
        """Test synchronizing all modalities together."""
        sync = TemporalSynchronizer(dim=512)
        
        vision = torch.randn(4, 8, 512)
        audio = torch.randn(4, 100, 512)
        proprio = torch.randn(4, 16, 512)
        
        v_sync, a_sync, p_sync = sync.synchronize_modalities(
            vision, audio, proprio, target_length=32
        )
        
        assert v_sync.shape == (4, 32, 512)
        assert a_sync.shape == (4, 32, 512)
        assert p_sync.shape == (4, 32, 512)


class TestCrossModalPredictor:
    """Test cross-modal prediction heads."""
    
    def test_predictor_directions(self):
        """Test all prediction directions."""
        predictor = CrossModalPredictor(dim=512)
        
        vision = torch.randn(4, 512)
        audio = torch.randn(4, 512)
        proprio = torch.randn(4, 512)
        
        # Test all directions
        v2a = predictor(vision, 'vision', 'audio')
        a2v = predictor(audio, 'audio', 'vision')
        v2p = predictor(vision, 'vision', 'proprio')
        p2v = predictor(proprio, 'proprio', 'vision')
        
        assert v2a.shape == (4, 512)
        assert a2v.shape == (4, 512)
        assert v2p.shape == (4, 512)
        assert p2v.shape == (4, 512)
    
    def test_prediction_loss(self):
        """Test prediction loss computation."""
        predictor = CrossModalPredictor(dim=512)
        
        vision = torch.randn(4, 512)
        audio = torch.randn(4, 512)
        proprio = torch.randn(4, 512)
        
        losses = predictor.compute_prediction_loss(vision, audio, proprio)
        
        assert 'loss_v2a' in losses
        assert 'loss_a2v' in losses
        assert 'loss_total' in losses
        assert losses['loss_total'].item() > 0


class TestFusionArchitectures:
    """Test different fusion architectures."""
    
    def test_early_fusion(self):
        """Test early fusion."""
        config = FusionConfig(dim=512)
        fusion = EarlyFusion(config)
        
        vision = torch.randn(4, 512)
        audio = torch.randn(4, 512)
        proprio = torch.randn(4, 512)
        
        output = fusion(vision, audio, proprio)
        
        assert output.shape == (4, 512)
    
    def test_mid_fusion(self):
        """Test mid-level cross-attention fusion."""
        config = FusionConfig(dim=512)
        fusion = MidFusion(config)
        
        vision = torch.randn(4, 8, 512)
        audio = torch.randn(4, 8, 512)
        proprio = torch.randn(4, 8, 512)
        
        fused, v_enr, a_enr, p_enr = fusion(vision, audio, proprio)
        
        assert fused.shape == (4, 512)
        assert v_enr.shape == (4, 8, 512)
        assert a_enr.shape == (4, 8, 512)
        assert p_enr.shape == (4, 8, 512)
    
    def test_late_fusion(self):
        """Test late fusion."""
        fusion = LateFusion(dim=512)
        
        vision = torch.randn(4, 512)
        audio = torch.randn(4, 512)
        proprio = torch.randn(4, 512)
        
        output = fusion(vision, audio, proprio)
        
        assert output.shape == (4, 512)
    
    def test_hierarchical_fusion(self):
        """Test hierarchical fusion combining all levels."""
        config = FusionConfig(dim=512, use_temporal_sync=False)
        fusion = HierarchicalFusion(config)
        
        vision = torch.randn(4, 8, 512)
        audio = torch.randn(4, 8, 512)
        proprio = torch.randn(4, 8, 512)
        
        output = fusion(vision, audio, proprio)
        
        assert output.shape == (4, 512)
    
    def test_hierarchical_all_levels(self):
        """Test hierarchical fusion returning all levels."""
        config = FusionConfig(dim=512, use_temporal_sync=False)
        fusion = HierarchicalFusion(config)
        
        vision = torch.randn(4, 8, 512)
        audio = torch.randn(4, 8, 512)
        proprio = torch.randn(4, 8, 512)
        
        result = fusion(vision, audio, proprio, return_all_levels=True)
        
        assert 'fused' in result
        assert 'early' in result
        assert 'mid' in result
        assert 'late' in result
        assert 'fusion_weights' in result


class TestEnhancedCrossModalFusion:
    """Test the complete enhanced fusion module."""
    
    def test_enhanced_fusion_forward(self):
        """Test enhanced fusion forward pass."""
        config = FusionConfig(
            dim=512,
            use_contrastive=True,
            use_cross_modal_prediction=True,
            use_temporal_sync=False,
        )
        fusion = EnhancedCrossModalFusion(config)
        
        vision = torch.randn(4, 8, 512)
        audio = torch.randn(4, 8, 512)
        proprio = torch.randn(4, 8, 512)
        
        result = fusion(vision, audio, proprio)
        
        assert 'fused' in result
        assert result['fused'].shape == (4, 512)
    
    def test_enhanced_fusion_with_losses(self):
        """Test enhanced fusion with loss computation."""
        config = FusionConfig(
            dim=512,
            use_contrastive=True,
            use_cross_modal_prediction=True,
            use_temporal_sync=False,
        )
        fusion = EnhancedCrossModalFusion(config)
        
        vision = torch.randn(4, 8, 512)
        audio = torch.randn(4, 8, 512)
        proprio = torch.randn(4, 8, 512)
        
        result = fusion(vision, audio, proprio, return_losses=True)
        
        assert 'contrastive' in result
        assert 'prediction' in result
        assert result['contrastive']['loss_total'].item() > 0
    
    def test_enhanced_fusion_legacy_mode(self):
        """Test legacy mode for backward compatibility."""
        config = FusionConfig(dim=512)
        fusion = EnhancedCrossModalFusion(config)
        
        vision = torch.randn(4, 8, 512)
        audio = torch.randn(4, 8, 512)
        proprio = torch.randn(4, 8, 512)
        
        result = fusion(vision, audio, proprio, use_legacy=True)
        
        assert 'fused' in result
        assert 'fused_sequence' in result
    
    def test_compute_loss_method(self):
        """Test the compute_loss convenience method."""
        config = FusionConfig(
            dim=512,
            use_contrastive=True,
            use_cross_modal_prediction=True,
        )
        fusion = EnhancedCrossModalFusion(config)
        
        vision = torch.randn(4, 8, 512)
        audio = torch.randn(4, 8, 512)
        proprio = torch.randn(4, 8, 512)
        
        losses = fusion.compute_loss(
            vision, audio, proprio,
            contrastive_weight=0.3,
            prediction_weight=0.25,
        )
        
        assert 'total_loss' in losses
        assert losses['total_loss'].item() > 0
    
    def test_get_modality_embeddings(self):
        """Test retrieving contrastive embeddings."""
        config = FusionConfig(dim=512, use_contrastive=True)
        fusion = EnhancedCrossModalFusion(config)
        
        vision = torch.randn(4, 8, 512)
        audio = torch.randn(4, 8, 512)
        proprio = torch.randn(4, 8, 512)
        
        embeddings = fusion.get_modality_embeddings(vision, audio, proprio)
        
        assert 'vision' in embeddings
        assert 'audio' in embeddings
        assert 'proprio' in embeddings


class TestCreateFusionModule:
    """Test the fusion module factory function."""
    
    def test_create_hierarchical(self):
        """Test creating hierarchical fusion."""
        fusion = create_fusion_module(
            fusion_type="hierarchical",
            dim=512,
            use_contrastive=True,
        )
        
        assert isinstance(fusion, EnhancedCrossModalFusion)
    
    def test_create_legacy(self):
        """Test creating legacy fusion."""
        fusion = create_fusion_module(
            fusion_type="legacy",
            dim=512,
        )
        
        assert isinstance(fusion, CrossModalFusion)
    
    def test_create_with_config(self):
        """Test creating fusion with full config."""
        config = FusionConfig(
            dim=256,
            num_heads=4,
            use_contrastive=True,
        )
        
        fusion = create_fusion_module(config=config)
        
        assert isinstance(fusion, EnhancedCrossModalFusion)


class TestGradientFlow:
    """Test gradient flow through fusion components."""
    
    def test_contrastive_gradients(self):
        """Test gradients flow through contrastive alignment."""
        align = ContrastiveAlignment(dim=512)
        
        vision = torch.randn(4, 512, requires_grad=True)
        audio = torch.randn(4, 512, requires_grad=True)
        proprio = torch.randn(4, 512, requires_grad=True)
        
        result = align(vision, audio, proprio)
        loss = result['loss_total']
        loss.backward()
        
        assert vision.grad is not None
        assert audio.grad is not None
        assert proprio.grad is not None
    
    def test_hierarchical_gradients(self):
        """Test gradients flow through hierarchical fusion."""
        config = FusionConfig(dim=512, use_temporal_sync=False)
        fusion = HierarchicalFusion(config)
        
        vision = torch.randn(4, 8, 512, requires_grad=True)
        audio = torch.randn(4, 8, 512, requires_grad=True)
        proprio = torch.randn(4, 8, 512, requires_grad=True)
        
        output = fusion(vision, audio, proprio)
        loss = output.sum()
        loss.backward()
        
        assert vision.grad is not None
        assert audio.grad is not None
        assert proprio.grad is not None
    
    def test_enhanced_fusion_gradients(self):
        """Test gradients flow through enhanced fusion."""
        config = FusionConfig(
            dim=512,
            use_contrastive=True,
            use_cross_modal_prediction=True,
            use_temporal_sync=False,
        )
        fusion = EnhancedCrossModalFusion(config)
        
        vision = torch.randn(4, 8, 512, requires_grad=True)
        audio = torch.randn(4, 8, 512, requires_grad=True)
        proprio = torch.randn(4, 8, 512, requires_grad=True)
        
        losses = fusion.compute_loss(vision, audio, proprio)
        losses['total_loss'].backward()
        
        assert vision.grad is not None
        assert audio.grad is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
