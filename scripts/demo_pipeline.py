#!/usr/bin/env python3
"""
NSCA Demo: Full cognitive pipeline demonstration.

Shows the complete perception -> understanding -> language pipeline:
1. Load video + audio
2. Encode through vision and audio encoders
3. Fuse multi-modal representations
4. Build world state
5. Generate language description
6. (Optional) Predict future states

Usage:
    python scripts/demo_pipeline.py --checkpoint checkpoints/extended_best.pth --data-dir /path/to/vis-data
    python scripts/demo_pipeline.py --checkpoint checkpoints/extended_best.pth --video /path/to/video.mp4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional
import random

import torch
import torch.nn.functional as F
import yaml

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.world_model.unified_world_model import UnifiedWorldModel, WorldModelConfig
from src.language.llm_integration import ConceptVerbalizer, LanguageConfig


def load_model(checkpoint_path: str, config: Dict, device: torch.device) -> UnifiedWorldModel:
    """Load model from checkpoint."""
    model_config = WorldModelConfig.from_dict(config['model'])
    model = UnifiedWorldModel(model_config).to(device)
    
    if checkpoint_path and Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded checkpoint: {checkpoint_path}")
        else:
            model.load_state_dict(checkpoint)
            print(f"Loaded checkpoint: {checkpoint_path}")
    
    model.eval()
    return model


def demo_single_sample(
    model: UnifiedWorldModel,
    video: torch.Tensor,
    audio: torch.Tensor,
    device: torch.device,
    sample_name: str = "Sample",
) -> Dict:
    """Run demo on a single video-audio pair."""
    
    print(f"\n{'='*60}")
    print(f"DEMO: {sample_name}")
    print(f"{'='*60}")
    
    # Move to device
    video = video.to(device)
    audio = audio.to(device)
    
    # Ensure batch dimension
    if video.dim() == 4:  # [T, C, H, W]
        video = video.unsqueeze(0)
    if audio.dim() == 1:  # [samples]
        audio = audio.unsqueeze(0)
    
    B, T, C, H, W = video.shape
    
    print(f"\n[1] INPUT")
    print(f"    Video: {T} frames, {H}x{W} pixels, {C} channels")
    print(f"    Audio: {audio.shape[-1]} samples ({audio.shape[-1]/16000:.2f} seconds)")
    
    # Perception
    model.eval()
    with torch.no_grad():
        # Encode video
        frames = video.view(B * T, C, H, W)
        vision_features = model.vision_encoder(frames)
        vision_features = vision_features.mean(dim=(2, 3))  # Global pool
        vision_features = vision_features.view(B, T, -1)
        
        print(f"\n[2] PERCEPTION (Encoding)")
        print(f"    Vision features: {vision_features.shape} (per-frame)")
        
        # Encode audio
        audio_features = model.audio_encoder(audio)
        print(f"    Audio features: {audio_features.shape}")
        
        # Cross-modal fusion
        proprio = torch.zeros(B, 1, 512, device=device)
        fused, _ = model.encode_multimodal(vision_features, audio_features.unsqueeze(1), proprio)
        
        print(f"\n[3] FUSION (Cross-modal integration)")
        print(f"    Fused representation: {fused.shape}")
        
        # Temporal processing -> World state
        world_state, temporal_features = model.temporal_model(fused)
        
        print(f"\n[4] WORLD STATE (Temporal understanding)")
        print(f"    World state: {world_state.shape}")
        print(f"    State magnitude: {world_state.norm().item():.2f}")
        
        # Similarity between video and audio (alignment score)
        video_emb = F.normalize(vision_features.mean(dim=1), dim=1)
        audio_emb = F.normalize(audio_features, dim=1)
        alignment = (video_emb * audio_emb).sum(dim=1).item()
        
        print(f"\n[5] CROSS-MODAL ALIGNMENT")
        print(f"    Video-Audio similarity: {alignment:.3f}")
        if alignment > 0.3:
            print(f"    -> Strong alignment: audio likely matches visual content")
        elif alignment > 0:
            print(f"    -> Weak alignment: some correspondence detected")
        else:
            print(f"    -> No alignment: audio may not match visual content")
        
        # Language description
        verbalizer = ConceptVerbalizer()
        
        # Generate pseudo-properties from world state
        # (In a full system, these would come from a property predictor)
        props = torch.sigmoid(world_state[0, :9])  # First 9 dims as properties
        description = verbalizer(props)[0]
        
        print(f"\n[6] LANGUAGE (Verbalization)")
        print(f"    Properties: {props.cpu().numpy().round(2)}")
        print(f"    Description: '{description}'")
        
        # Dynamics prediction (imagination)
        print(f"\n[7] IMAGINATION (Future prediction)")
        try:
            # Action dim should match model's expected action_dim (default 32)
            action_dim = model.dynamics.config.action_dim if hasattr(model.dynamics, 'config') else 32
            dummy_actions = torch.randn(B, 3, action_dim, device=device) * 0.1
            predicted_states, uncertainties = model.imagine(world_state, dummy_actions)
            print(f"    Predicted {predicted_states.shape[1]} future states")
            print(f"    Uncertainty: {uncertainties.mean().item():.3f}")
        except Exception as e:
            print(f"    Imagination module: {type(model.dynamics).__name__}")
            print(f"    (Dynamics prediction skipped - requires action training)")
        
    return {
        'world_state': world_state,
        'alignment': alignment,
        'description': description,
        'properties': props.cpu().numpy(),
    }


def run_demo_with_dataset(
    model: UnifiedWorldModel,
    data_dir: str,
    device: torch.device,
    num_samples: int = 3,
):
    """Run demo on samples from the dataset."""
    from train_multimodal import GreatestHitsDataset
    
    dataset = GreatestHitsDataset(
        data_dir=data_dir,
        split='val',
        n_frames=8,
        audio_duration=1.0,
        augment=False
    )
    
    print(f"\nLoaded {len(dataset)} samples from {data_dir}")
    
    # Random samples
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    
    results = []
    for i, idx in enumerate(indices):
        sample = dataset[idx]
        result = demo_single_sample(
            model=model,
            video=sample['video'],
            audio=sample['audio'],
            device=device,
            sample_name=f"Sample {i+1} (index {idx})"
        )
        results.append(result)
    
    # Summary
    print(f"\n{'='*60}")
    print("DEMO SUMMARY")
    print(f"{'='*60}")
    
    avg_alignment = sum(r['alignment'] for r in results) / len(results)
    print(f"\nProcessed {len(results)} samples")
    print(f"Average cross-modal alignment: {avg_alignment:.3f}")
    
    print("\nDescriptions generated:")
    for i, r in enumerate(results):
        print(f"  {i+1}. {r['description']}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='NSCA Demo Pipeline')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint')
    parser.add_argument('--config', type=str, default='configs/training_config.yaml')
    parser.add_argument('--data-dir', type=str, help='Path to GreatestHits data')
    parser.add_argument('--video', type=str, help='Path to single video file')
    parser.add_argument('--num-samples', type=int, default=3, help='Number of samples to demo')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Load model
    print("\nLoading model...")
    model = load_model(args.checkpoint, config, device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    if args.data_dir:
        # Demo with dataset
        run_demo_with_dataset(model, args.data_dir, device, args.num_samples)
    elif args.video:
        print("Single video demo not yet implemented - use --data-dir")
    else:
        print("Please provide --data-dir or --video")
        return
    
    print("\n" + "="*60)
    print("DEMO COMPLETE")
    print("="*60)
    print("\nThe NSCA pipeline processes multimodal input through:")
    print("  1. Perception: Vision + Audio encoding with innate priors")
    print("  2. Fusion: Cross-modal attention to align modalities")
    print("  3. World State: Temporal processing to build understanding")
    print("  4. Language: Property-based verbalization")
    print("  5. Imagination: Dynamics prediction for future states")


if __name__ == '__main__':
    main()
