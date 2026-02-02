#!/usr/bin/env python3
"""
Test the trained multi-modal model.

Evaluates:
1. Cross-modal retrieval (can it match video to correct audio?)
2. Feature visualization
3. Sample predictions

Usage:
    python scripts/test_multimodal.py --checkpoint checkpoints/multimodal_best.pth --data-dir /path/to/vis-data
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Add project root
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from scripts.train_multimodal import GreatestHitsDataset, CrossModalWorldModel


def test_retrieval(model, dataloader, device, k_values=[1, 5, 10]):
    """
    Test cross-modal retrieval accuracy.
    
    Given a video, can we find the matching audio?
    Given an audio, can we find the matching video?
    """
    print("\n" + "=" * 60)
    print("CROSS-MODAL RETRIEVAL TEST")
    print("=" * 60)
    
    model.eval()
    
    all_vision_features = []
    all_audio_features = []
    all_names = []
    
    with torch.no_grad():
        for batch in dataloader:
            video = batch['video'].to(device)
            audio = batch['audio'].to(device)
            names = batch['name']
            
            outputs = model(video, audio)
            
            # Get normalized projections (used for contrastive learning)
            vision_proj = outputs['vision_proj']  # Already normalized
            audio_proj = outputs['audio_proj']
            
            all_vision_features.append(vision_proj.cpu())
            all_audio_features.append(audio_proj.cpu())
            all_names.extend(names)
    
    # Stack all features
    vision_features = torch.cat(all_vision_features, dim=0)  # [N, 128]
    audio_features = torch.cat(all_audio_features, dim=0)    # [N, 128]
    
    N = vision_features.shape[0]
    print(f"\nTotal samples: {N}")
    
    # Compute similarity matrix
    similarity = torch.mm(vision_features, audio_features.T)  # [N, N]
    
    # Video → Audio retrieval
    print("\n--- Video → Audio Retrieval ---")
    v2a_ranks = []
    for i in range(N):
        # For video i, rank all audios by similarity
        sims = similarity[i]
        sorted_indices = torch.argsort(sims, descending=True)
        rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item() + 1
        v2a_ranks.append(rank)
    
    v2a_ranks = np.array(v2a_ranks)
    for k in k_values:
        recall_at_k = (v2a_ranks <= k).mean() * 100
        print(f"  Recall@{k}: {recall_at_k:.1f}%")
    print(f"  Median Rank: {np.median(v2a_ranks):.1f}")
    print(f"  Mean Rank: {np.mean(v2a_ranks):.1f}")
    
    # Audio → Video retrieval
    print("\n--- Audio → Video Retrieval ---")
    a2v_ranks = []
    for i in range(N):
        # For audio i, rank all videos by similarity
        sims = similarity[:, i]
        sorted_indices = torch.argsort(sims, descending=True)
        rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item() + 1
        a2v_ranks.append(rank)
    
    a2v_ranks = np.array(a2v_ranks)
    for k in k_values:
        recall_at_k = (a2v_ranks <= k).mean() * 100
        print(f"  Recall@{k}: {recall_at_k:.1f}%")
    print(f"  Median Rank: {np.median(a2v_ranks):.1f}")
    print(f"  Mean Rank: {np.mean(a2v_ranks):.1f}")
    
    # Random baseline
    print("\n--- Random Baseline ---")
    print(f"  Recall@1: {100/N:.1f}%")
    print(f"  Recall@5: {500/N:.1f}%")
    print(f"  Recall@10: {1000/N:.1f}%")
    print(f"  Expected Median Rank: {N/2:.1f}")
    
    return {
        'v2a_recall@1': (v2a_ranks <= 1).mean() * 100,
        'v2a_recall@5': (v2a_ranks <= 5).mean() * 100,
        'a2v_recall@1': (a2v_ranks <= 1).mean() * 100,
        'a2v_recall@5': (a2v_ranks <= 5).mean() * 100,
    }


def test_cross_modal_prediction(model, dataloader, device, n_samples=5):
    """
    Test cross-modal prediction quality.
    
    Can the model predict audio features from video and vice versa?
    """
    print("\n" + "=" * 60)
    print("CROSS-MODAL PREDICTION TEST")
    print("=" * 60)
    
    model.eval()
    
    total_v2a_error = 0
    total_a2v_error = 0
    n_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            video = batch['video'].to(device)
            audio = batch['audio'].to(device)
            
            outputs = model(video, audio)
            
            # Compute prediction errors
            v2a_error = F.mse_loss(outputs['vision_to_audio_pred'], outputs['audio_features'])
            a2v_error = F.mse_loss(outputs['audio_to_vision_pred'], outputs['vision_features'])
            
            total_v2a_error += v2a_error.item()
            total_a2v_error += a2v_error.item()
            n_batches += 1
    
    avg_v2a_error = total_v2a_error / n_batches
    avg_a2v_error = total_a2v_error / n_batches
    
    print(f"\nVideo → Audio prediction MSE: {avg_v2a_error:.4f}")
    print(f"Audio → Video prediction MSE: {avg_a2v_error:.4f}")
    
    # Compare to random baseline (predicting zeros)
    print("\n(Lower is better. Random baseline would be ~1.0 for normalized features)")
    
    return {'v2a_mse': avg_v2a_error, 'a2v_mse': avg_a2v_error}


def test_feature_quality(model, dataloader, device):
    """
    Test the quality of learned features.
    """
    print("\n" + "=" * 60)
    print("FEATURE QUALITY TEST")
    print("=" * 60)
    
    model.eval()
    
    all_vision = []
    all_audio = []
    
    with torch.no_grad():
        for batch in dataloader:
            video = batch['video'].to(device)
            audio = batch['audio'].to(device)
            
            outputs = model(video, audio)
            
            all_vision.append(outputs['vision_features'].cpu())
            all_audio.append(outputs['audio_features'].cpu())
    
    vision = torch.cat(all_vision, dim=0)
    audio = torch.cat(all_audio, dim=0)
    
    # Check feature statistics
    print(f"\nVision features:")
    print(f"  Mean: {vision.mean().item():.4f}")
    print(f"  Std: {vision.std().item():.4f}")
    print(f"  Min: {vision.min().item():.4f}")
    print(f"  Max: {vision.max().item():.4f}")
    
    print(f"\nAudio features:")
    print(f"  Mean: {audio.mean().item():.4f}")
    print(f"  Std: {audio.std().item():.4f}")
    print(f"  Min: {audio.min().item():.4f}")
    print(f"  Max: {audio.max().item():.4f}")
    
    # Check if features are diverse (not collapsed)
    vision_var = vision.var(dim=0).mean().item()
    audio_var = audio.var(dim=0).mean().item()
    
    print(f"\nFeature diversity (variance across samples):")
    print(f"  Vision: {vision_var:.4f}")
    print(f"  Audio: {audio_var:.4f}")
    
    if vision_var < 0.01 or audio_var < 0.01:
        print("  WARNING: Features may have collapsed!")
    else:
        print("  Features are diverse (good!)")
    
    # Check cross-modal correlation
    # Flatten and compute correlation
    v_flat = vision.flatten()
    a_flat = audio.flatten()[:len(v_flat)]
    correlation = torch.corrcoef(torch.stack([v_flat, a_flat]))[0, 1].item()
    print(f"\nCross-modal correlation: {correlation:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Test multi-modal model")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to checkpoint")
    parser.add_argument('--data-dir', type=str, required=True, help="Path to data")
    parser.add_argument('--device', type=str, default='cuda', help="Device")
    parser.add_argument('--batch-size', type=int, default=16, help="Batch size")
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load model
    print(f"\nLoading model from: {args.checkpoint}")
    model = CrossModalWorldModel()
    
    state_dict = torch.load(args.checkpoint, map_location=device)
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded! Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Load validation data
    print(f"\nLoading data from: {args.data_dir}")
    val_dataset = GreatestHitsDataset(args.data_dir, split='val')
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
    )
    
    # Run tests
    test_feature_quality(model, val_loader, device)
    retrieval_results = test_retrieval(model, val_loader, device)
    prediction_results = test_cross_modal_prediction(model, val_loader, device)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nCross-Modal Retrieval (higher is better):")
    print(f"  Video→Audio Recall@1: {retrieval_results['v2a_recall@1']:.1f}%")
    print(f"  Audio→Video Recall@1: {retrieval_results['a2v_recall@1']:.1f}%")
    
    print(f"\nCross-Modal Prediction (lower is better):")
    print(f"  Video→Audio MSE: {prediction_results['v2a_mse']:.4f}")
    print(f"  Audio→Video MSE: {prediction_results['a2v_mse']:.4f}")
    
    # Verdict
    print("\n" + "=" * 60)
    if retrieval_results['v2a_recall@1'] > 5 and retrieval_results['a2v_recall@1'] > 5:
        print("VERDICT: Model learned meaningful cross-modal correspondence!")
        print("The model can match videos with their sounds better than random.")
    else:
        print("VERDICT: Model needs more training or data augmentation.")
        print("Retrieval is close to random chance.")
    print("=" * 60)


if __name__ == "__main__":
    main()
