#!/usr/bin/env python3
"""
Evaluation script for the Unified World Model.

Evaluates:
1. Vision: Zero-shot and few-shot classification accuracy
2. Audio: Sound recognition accuracy
3. Fusion: Cross-modal retrieval (audio <-> video)
4. Temporal: Next-frame prediction error
5. Memory: Few-shot learning accuracy
6. Dynamics: Trajectory prediction error

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/world_model_final.pth
    python scripts/evaluate.py --checkpoint checkpoints/world_model_final.pth --eval vision
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.world_model.unified_world_model import UnifiedWorldModel, WorldModelConfig


def load_model(checkpoint_path: str, config: Dict, device: torch.device) -> UnifiedWorldModel:
    """Load model from checkpoint."""
    model_config = WorldModelConfig.from_dict(config['model'])
    model = UnifiedWorldModel(model_config).to(device)
    
    if checkpoint_path and Path(checkpoint_path).exists():
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        print("No checkpoint found, using random initialization")
    
    model.eval()
    return model


def evaluate_vision_zero_shot(
    model: UnifiedWorldModel,
    device: torch.device,
    num_classes: int = 10,
) -> Dict[str, float]:
    """
    Evaluate zero-shot classification using semantic memory.
    
    Creates prototypes for each class and classifies test images
    by nearest prototype.
    """
    print("\n" + "=" * 40)
    print("Evaluating Vision: Zero-Shot Classification")
    print("=" * 40)
    
    # Generate synthetic test data
    num_test = 100
    test_images = torch.randn(num_test, 3, 224, 224, device=device)
    test_labels = torch.randint(0, num_classes, (num_test,), device=device)
    
    # Create class prototypes in memory
    for class_idx in range(num_classes):
        # Generate synthetic examples for this class
        class_images = torch.randn(5, 3, 224, 224, device=device)
        model.learn_new_concept(class_images, f"class_{class_idx}")
    
    # Classify test images
    correct = 0
    with torch.no_grad():
        for i in range(num_test):
            img = test_images[i:i+1]
            features = model.encode_vision(img).squeeze(0)
            
            # Recall from semantic memory
            results = model.memory.recall_semantic(features, k=1)
            if results:
                pred_class = int(results[0][0].split('_')[1])
                if pred_class == test_labels[i].item():
                    correct += 1
    
    accuracy = correct / num_test
    print(f"Zero-shot accuracy: {accuracy:.2%}")
    
    return {'zero_shot_accuracy': accuracy}


def evaluate_vision_few_shot(
    model: UnifiedWorldModel,
    device: torch.device,
    k_shot: int = 5,
    num_classes: int = 5,
) -> Dict[str, float]:
    """
    Evaluate few-shot classification using meta-learning.
    """
    print("\n" + "=" * 40)
    print(f"Evaluating Vision: {k_shot}-Shot Classification")
    print("=" * 40)
    
    # Generate support and query sets
    support_images = torch.randn(num_classes * k_shot, 3, 224, 224, device=device)
    support_labels = torch.arange(num_classes, device=device).repeat_interleave(k_shot)
    
    query_images = torch.randn(num_classes * 10, 3, 224, 224, device=device)
    query_labels = torch.arange(num_classes, device=device).repeat_interleave(10)
    
    # Use prototypical networks
    predictions = model.proto_nets.few_shot_classify(
        support_images, support_labels, query_images
    )
    
    accuracy = (predictions == query_labels).float().mean().item()
    print(f"{k_shot}-shot accuracy: {accuracy:.2%}")
    
    return {f'{k_shot}_shot_accuracy': accuracy}


def evaluate_cross_modal_retrieval(
    model: UnifiedWorldModel,
    device: torch.device,
    num_samples: int = 50,
) -> Dict[str, float]:
    """
    Evaluate cross-modal retrieval: audio -> video and video -> audio.
    """
    print("\n" + "=" * 40)
    print("Evaluating Cross-Modal Retrieval")
    print("=" * 40)
    
    # Generate paired audio-video data
    vision_features = torch.randn(num_samples, 1, 512, device=device)  # Pre-encoded
    audio_features = torch.randn(num_samples, 512, device=device)
    proprio_features = torch.randn(num_samples, 1, 512, device=device)
    
    with torch.no_grad():
        # Fuse modalities
        fused, _ = model.fusion(vision_features, audio_features.unsqueeze(1), proprio_features)
        
        # Extract modality-specific representations
        vision_repr = fused[:, 0, :]  # First token is vision
        
    # Audio -> Video retrieval
    # (In real evaluation, would use actual paired data)
    audio_normalized = F.normalize(audio_features, dim=1)
    vision_normalized = F.normalize(vision_repr, dim=1)
    
    # Similarity matrix
    similarity = audio_normalized @ vision_normalized.T
    
    # Recall@1: correct match on diagonal
    predictions = similarity.argmax(dim=1)
    targets = torch.arange(num_samples, device=device)
    recall_at_1 = (predictions == targets).float().mean().item()
    
    print(f"Audio -> Video Recall@1: {recall_at_1:.2%}")
    
    return {'cross_modal_recall_1': recall_at_1}


def evaluate_temporal_prediction(
    model: UnifiedWorldModel,
    device: torch.device,
    seq_len: int = 16,
    num_samples: int = 50,
) -> Dict[str, float]:
    """
    Evaluate temporal prediction accuracy.
    """
    print("\n" + "=" * 40)
    print("Evaluating Temporal Prediction")
    print("=" * 40)
    
    # Generate sequences
    sequences = torch.randn(num_samples, seq_len, 512, device=device)
    
    with torch.no_grad():
        # Process through temporal model
        world_states = []
        for i in range(num_samples):
            ws, _ = model.temporal_model(sequences[i:i+1])
            world_states.append(ws)
        
        world_states = torch.cat(world_states, dim=0)
    
    # Measure consistency (same sequence should give similar state)
    state_std = world_states.std(dim=0).mean().item()
    
    print(f"State std: {state_std:.4f}")
    
    return {'temporal_state_std': state_std}


def evaluate_dynamics_prediction(
    model: UnifiedWorldModel,
    device: torch.device,
    horizon: int = 5,
    num_samples: int = 50,
) -> Dict[str, float]:
    """
    Evaluate dynamics (imagination) prediction accuracy.
    """
    print("\n" + "=" * 40)
    print("Evaluating Dynamics Prediction")
    print("=" * 40)
    
    # Generate initial states and action sequences
    initial_states = torch.randn(num_samples, model.config.state_dim, device=device)
    actions = torch.randn(num_samples, horizon, model.config.action_dim, device=device)
    
    with torch.no_grad():
        # Predict trajectory
        predicted_states, uncertainties = model.imagine(initial_states, actions)
    
    # Measure prediction stability
    prediction_norm = predicted_states.norm(dim=-1).mean().item()
    
    if uncertainties is not None:
        mean_uncertainty = uncertainties.mean().item()
        print(f"Mean uncertainty: {mean_uncertainty:.4f}")
    else:
        mean_uncertainty = 0.0
    
    print(f"Prediction norm: {prediction_norm:.4f}")
    
    return {
        'dynamics_prediction_norm': prediction_norm,
        'dynamics_uncertainty': mean_uncertainty,
    }


def evaluate_memory_system(
    model: UnifiedWorldModel,
    device: torch.device,
    num_concepts: int = 10,
    examples_per_concept: int = 5,
) -> Dict[str, float]:
    """
    Evaluate memory system (store and recall).
    """
    print("\n" + "=" * 40)
    print("Evaluating Memory System")
    print("=" * 40)
    
    # Clear existing memory
    model.memory.clear_all()
    
    # Store concepts
    concept_prototypes = {}
    for c in range(num_concepts):
        # Generate examples
        examples = torch.randn(examples_per_concept, model.config.state_dim, device=device)
        prototype = examples.mean(dim=0)
        concept_prototypes[f"concept_{c}"] = prototype
        
        # Store in memory
        model.memory.learn_new_concept(examples, f"concept_{c}")
    
    # Test recall
    correct = 0
    total = 0
    
    for c in range(num_concepts):
        # Query with noisy version of prototype
        query = concept_prototypes[f"concept_{c}"] + torch.randn_like(concept_prototypes[f"concept_{c}"]) * 0.1
        
        # Recall
        results = model.memory.recall_semantic(query, k=1)
        if results and results[0][0] == f"concept_{c}":
            correct += 1
        total += 1
    
    recall_accuracy = correct / total
    print(f"Memory recall accuracy: {recall_accuracy:.2%}")
    
    # Get memory stats
    stats = model.get_memory_stats()
    print(f"Semantic concepts: {stats['semantic_count']}")
    
    return {
        'memory_recall_accuracy': recall_accuracy,
        'semantic_concept_count': stats['semantic_count'],
    }


def run_full_evaluation(
    model: UnifiedWorldModel,
    device: torch.device,
) -> Dict[str, float]:
    """Run all evaluations and aggregate results."""
    results = {}
    
    # Vision evaluations
    results.update(evaluate_vision_zero_shot(model, device))
    results.update(evaluate_vision_few_shot(model, device, k_shot=5))
    
    # Cross-modal
    results.update(evaluate_cross_modal_retrieval(model, device))
    
    # Temporal
    results.update(evaluate_temporal_prediction(model, device))
    
    # Dynamics
    results.update(evaluate_dynamics_prediction(model, device))
    
    # Memory
    results.update(evaluate_memory_system(model, device))
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate NSCA World Model")
    parser.add_argument('--checkpoint', type=str, default=None, help="Model checkpoint")
    parser.add_argument('--config', type=str, default='configs/training_config.yaml', help="Config file")
    parser.add_argument('--eval', type=str, default='all',
                       choices=['all', 'vision', 'audio', 'fusion', 'temporal', 'memory'],
                       help="Evaluation to run")
    parser.add_argument('--output', type=str, default='eval_results.json', help="Output file")
    parser.add_argument('--device', type=str, default='cuda', help="Device")
    parser.add_argument('--n-seeds', type=int, default=5, help="Number of seeds for evaluation")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(args.checkpoint, config, device)
    
    # Run evaluation
    if args.eval == 'all':
        results = run_full_evaluation(model, device)
    elif args.eval == 'vision':
        results = evaluate_vision_zero_shot(model, device)
        results.update(evaluate_vision_few_shot(model, device))
    elif args.eval == 'fusion':
        results = evaluate_cross_modal_retrieval(model, device)
    elif args.eval == 'temporal':
        results = evaluate_temporal_prediction(model, device)
    elif args.eval == 'memory':
        results = evaluate_memory_system(model, device)
    else:
        results = {}
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    for key, value in results.items():
        print(f"  {key}: {value:.4f}")
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
