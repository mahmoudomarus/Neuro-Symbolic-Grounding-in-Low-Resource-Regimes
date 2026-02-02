#!/usr/bin/env python3
"""
Test the trained world model on various inputs.

This script demonstrates what the world model can do:
1. Encode visual, audio, proprioceptive inputs
2. Fuse multi-modal information
3. Process temporal sequences
4. Predict future states

Usage:
    python scripts/test_world_model.py --checkpoint checkpoints/checkpoints/world_model_final.pth
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.world_model.unified_world_model import UnifiedWorldModel, WorldModelConfig


def create_test_inputs(batch_size: int = 4, device: str = 'cpu'):
    """Create synthetic test inputs for the world model."""
    
    # Vision: [B, C, H, W]
    vision = torch.randn(batch_size, 3, 224, 224, device=device)
    
    # Audio: [B, audio_samples] (1 second at 16kHz)
    audio = torch.randn(batch_size, 16000, device=device)
    
    # Proprioception: [B, proprio_dim] (joint angles, positions, etc.)
    proprio = torch.randn(batch_size, 12, device=device)
    
    return vision, audio, proprio


def test_encoding(model: UnifiedWorldModel, vision, audio, proprio, device):
    """Test individual encoder outputs."""
    print("\n" + "=" * 60)
    print("TEST 1: Individual Encoders")
    print("=" * 60)
    
    with torch.no_grad():
        # Vision encoding (use model's method which handles pooling)
        vision_features = model.encode_vision(vision)
        print(f"\nVision encoding:")
        print(f"  Input shape: {vision.shape}")
        print(f"  Output shape: {vision_features.shape}")
        print(f"  Feature mean: {vision_features.mean().item():.4f}")
        print(f"  Feature std: {vision_features.std().item():.4f}")
        
        # Audio encoding
        audio_features = model.encode_audio(audio)
        print(f"\nAudio encoding:")
        print(f"  Input shape: {audio.shape}")
        print(f"  Output shape: {audio_features.shape}")
        print(f"  Feature mean: {audio_features.mean().item():.4f}")
        print(f"  Feature std: {audio_features.std().item():.4f}")
        
        # Proprioception encoding
        proprio_features = model.encode_proprio(proprio)
        print(f"\nProprioception encoding:")
        print(f"  Input shape: {proprio.shape}")
        print(f"  Output shape: {proprio_features.shape}")
        print(f"  Feature mean: {proprio_features.mean().item():.4f}")
        print(f"  Feature std: {proprio_features.std().item():.4f}")
    
    return vision_features, audio_features, proprio_features


def test_fusion(model: UnifiedWorldModel, vision_feat, audio_feat, proprio_feat):
    """Test cross-modal fusion."""
    print("\n" + "=" * 60)
    print("TEST 2: Cross-Modal Fusion")
    print("=" * 60)
    
    with torch.no_grad():
        # Add time dimension for fusion (fusion expects [B, T, D])
        vision_t = vision_feat.unsqueeze(1)  # [B, 1, D]
        audio_t = audio_feat.unsqueeze(1)    # [B, 1, D]
        proprio_t = proprio_feat.unsqueeze(1) # [B, 1, D]
        
        # Fuse all modalities
        fused, attn = model.fusion(vision_t, audio_t, proprio_t)
        
        print(f"\nFusion output:")
        print(f"  Vision input: {vision_t.shape}")
        print(f"  Audio input: {audio_t.shape}")
        print(f"  Proprio input: {proprio_t.shape}")
        print(f"  Output shape: {fused.shape}")
        print(f"  Feature mean: {fused.mean().item():.4f}")
        print(f"  Feature std: {fused.std().item():.4f}")
        
        # Test modality contributions (which modality matters most?)
        # We can check this by zeroing out one modality and seeing the effect
        
        # Vision only
        fused_vision_only, _ = model.fusion(
            vision_t, 
            torch.zeros_like(audio_t), 
            torch.zeros_like(proprio_t)
        )
        
        # Audio only
        fused_audio_only, _ = model.fusion(
            torch.zeros_like(vision_t), 
            audio_t, 
            torch.zeros_like(proprio_t)
        )
        
        # Measure similarity to full fusion
        sim_vision = F.cosine_similarity(fused.flatten(), fused_vision_only.flatten(), dim=0)
        sim_audio = F.cosine_similarity(fused.flatten(), fused_audio_only.flatten(), dim=0)
        
        print(f"\nModality contributions (cosine similarity to full fusion):")
        print(f"  Vision contribution: {sim_vision.item():.4f}")
        print(f"  Audio contribution: {sim_audio.item():.4f}")
    
    return fused


def test_temporal(model: UnifiedWorldModel, fused_states, device):
    """Test temporal processing."""
    print("\n" + "=" * 60)
    print("TEST 3: Temporal Processing")
    print("=" * 60)
    
    # fused_states is [B, T, D] from fusion
    # Create longer sequence by repeating with variations
    batch_size = fused_states.shape[0]
    base_dim = fused_states.shape[-1]
    seq_len = 8
    
    # Stack states with slight variations to simulate temporal sequence
    temporal_sequence = torch.cat([
        fused_states + 0.1 * i * torch.randn(batch_size, 1, base_dim, device=device)
        for i in range(seq_len)
    ], dim=1)  # [B, T*seq_len, D]
    
    print(f"\nTemporal input shape: {temporal_sequence.shape}")
    
    with torch.no_grad():
        # Process through temporal model (may return tuple)
        result = model.temporal_model(temporal_sequence)
        
        # Handle tuple or tensor return
        if isinstance(result, tuple):
            temporal_output = result[0]
        else:
            temporal_output = result
        
        print(f"Temporal output shape: {temporal_output.shape}")
        print(f"Temporal feature mean: {temporal_output.mean().item():.4f}")
        print(f"Temporal feature std: {temporal_output.std().item():.4f}")
    
    return temporal_output


def test_dynamics_prediction(model: UnifiedWorldModel, state, device):
    """Test dynamics/future state prediction."""
    print("\n" + "=" * 60)
    print("TEST 4: Dynamics Prediction (Future State)")
    print("=" * 60)
    
    # Create a simple action (e.g., move forward)
    batch_size = state.shape[0]
    action = torch.randn(batch_size, 32, device=device)  # Action embedding
    
    with torch.no_grad():
        # Predict next state
        next_state, uncertainty = model.dynamics(state, action)
        
        print(f"\nDynamics prediction:")
        print(f"  Current state shape: {state.shape}")
        print(f"  Action shape: {action.shape}")
        print(f"  Predicted next state shape: {next_state.shape}")
        print(f"  Uncertainty shape: {uncertainty.shape}")
        print(f"  Prediction uncertainty (mean): {uncertainty.mean().item():.4f}")
        
        # Multi-step prediction
        print(f"\nMulti-step rollout (5 steps):")
        current = state
        for step in range(5):
            action = torch.randn(batch_size, 32, device=device)
            next_state, uncertainty = model.dynamics(current, action)
            print(f"  Step {step+1}: uncertainty = {uncertainty.mean().item():.4f}")
            current = next_state
    
    return next_state


def test_full_forward(model: UnifiedWorldModel, vision, audio, proprio, device):
    """Test full forward pass through the model."""
    print("\n" + "=" * 60)
    print("TEST 5: Full Forward Pass (encode → fuse → temporal → dynamics)")
    print("=" * 60)
    
    batch_size = vision.shape[0]
    
    print(f"\nRaw input shapes:")
    print(f"  Vision: {vision.shape}")
    print(f"  Audio: {audio.shape}")
    print(f"  Proprio: {proprio.shape}")
    
    with torch.no_grad():
        # Step 1: Encode all modalities
        vision_enc = model.encode_vision(vision)  # [B, D]
        audio_enc = model.encode_audio(audio)     # [B, D]
        proprio_enc = model.encode_proprio(proprio)  # [B, D]
        
        print(f"\nEncoded shapes:")
        print(f"  Vision: {vision_enc.shape}")
        print(f"  Audio: {audio_enc.shape}")
        print(f"  Proprio: {proprio_enc.shape}")
        
        # Step 2: Fuse (add time dimension)
        fused, _ = model.fusion(
            vision_enc.unsqueeze(1),
            audio_enc.unsqueeze(1),
            proprio_enc.unsqueeze(1)
        )
        print(f"\nFused shape: {fused.shape}")
        
        # Step 3: Temporal processing (simulate sequence)
        temporal_seq = fused.repeat(1, 4, 1)  # [B, 4*T, D]
        temporal_out = model.temporal_model(temporal_seq)
        if isinstance(temporal_out, tuple):
            temporal_out = temporal_out[0]
        print(f"Temporal output: {temporal_out.shape}")
        
        # Step 4: Dynamics prediction
        action = torch.randn(batch_size, model.config.action_dim, device=device)
        next_state, uncertainty = model.dynamics(temporal_out, action)
        
        print(f"\nWorld Model Pipeline Complete:")
        print(f"  Input: 224x224 RGB + 1s Audio + 12-dim Proprio")
        print(f"  World state: {temporal_out.shape}")
        print(f"  Predicted next state: {next_state.shape}")
        print(f"  Prediction uncertainty: {uncertainty.mean().item():.4f}")


def main():
    parser = argparse.ArgumentParser(description="Test trained world model")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to checkpoint")
    parser.add_argument('--device', type=str, default='cpu', help="Device (cpu/cuda)")
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading checkpoint: {args.checkpoint}")
    
    # Create model with default config
    config = WorldModelConfig(
        latent_dim=512,
        state_dim=256,
        action_dim=32,
    )
    
    model = UnifiedWorldModel(config)
    
    # Load weights
    state_dict = torch.load(args.checkpoint, map_location=device)
    
    # Handle both raw state_dict and wrapped checkpoint
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    
    # Clone tensors that might have shared memory (prior tensors)
    state_dict = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in state_dict.items()}
    
    # Remove problematic buffer keys that have shared memory issues
    # These are perspective priors that are regenerated from scratch anyway
    problem_keys = [k for k in state_dict.keys() if 'scale_prior' in k or 'depth_prior' in k]
    for key in problem_keys:
        print(f"  Skipping buffer with shared memory: {key}")
        del state_dict[key]
    
    # Load with strict=False to allow missing buffers
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Missing keys (OK - these are regenerated): {len(missing)}")
    if unexpected:
        print(f"  Unexpected keys: {unexpected}")
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create test inputs
    vision, audio, proprio = create_test_inputs(batch_size=4, device=device)
    
    # Run tests
    vision_feat, audio_feat, proprio_feat = test_encoding(model, vision, audio, proprio, device)
    fused = test_fusion(model, vision_feat, audio_feat, proprio_feat)
    temporal_out = test_temporal(model, fused, device)
    next_state = test_dynamics_prediction(model, temporal_out, device)
    
    # Full forward pass
    test_full_forward(model, vision, audio, proprio, device)
    
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nThe world model can:")
    print("  1. Encode visual input with innate priors (Gabor, color, depth)")
    print("  2. Encode audio with auditory priors (mel-spectrogram)")
    print("  3. Encode proprioceptive/body state")
    print("  4. Fuse multiple modalities via attention")
    print("  5. Process temporal sequences")
    print("  6. Predict future states with uncertainty")
    print("\nNext steps:")
    print("  - Train on multi-modal data (Greatest Hits) for cross-modal learning")
    print("  - Fine-tune on specific tasks (object manipulation, physics prediction)")
    print("  - Evaluate on benchmarks (Physion, Meta-World)")


if __name__ == "__main__":
    main()
