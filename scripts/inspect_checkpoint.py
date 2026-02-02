#!/usr/bin/env python3
"""
Inspect a trained checkpoint to understand what it contains.

Usage:
    python scripts/inspect_checkpoint.py checkpoints/checkpoints/world_model_final.pth
"""

import argparse
import sys
from pathlib import Path

import torch


def inspect_checkpoint(checkpoint_path: str):
    """Load and inspect a checkpoint file."""
    print("=" * 60)
    print(f"CHECKPOINT INSPECTION: {checkpoint_path}")
    print("=" * 60)
    
    # Check file exists
    if not Path(checkpoint_path).exists():
        print(f"ERROR: File not found: {checkpoint_path}")
        return
    
    # Load checkpoint
    print("\nLoading checkpoint...")
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    except Exception as e:
        print(f"ERROR loading checkpoint: {e}")
        return
    
    # Check what type of object it is
    print(f"\nCheckpoint type: {type(checkpoint)}")
    
    if isinstance(checkpoint, dict):
        print(f"\nKeys in checkpoint: {list(checkpoint.keys())}")
        
        # Check for common keys
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print(f"\n--- Model State Dict ---")
            print(f"Number of parameters: {len(state_dict)}")
            
            # List first 20 parameter names
            print("\nFirst 20 parameters:")
            for i, (name, tensor) in enumerate(state_dict.items()):
                if i >= 20:
                    print(f"  ... and {len(state_dict) - 20} more")
                    break
                print(f"  {name}: {tensor.shape}")
            
            # Total parameters
            total_params = sum(t.numel() for t in state_dict.values())
            print(f"\nTotal parameters: {total_params:,}")
        
        if 'epoch' in checkpoint:
            print(f"\nEpoch: {checkpoint['epoch']}")
        
        if 'train_metrics' in checkpoint:
            print(f"\nTrain metrics: {checkpoint['train_metrics']}")
        
        if 'val_metrics' in checkpoint:
            print(f"\nVal metrics: {checkpoint['val_metrics']}")
        
        if 'config' in checkpoint:
            print(f"\nConfig: {checkpoint['config']}")
            
    elif isinstance(checkpoint, torch.nn.Module):
        print("Checkpoint is a full model object")
        
    else:
        # It might be a raw state_dict
        if hasattr(checkpoint, 'keys'):
            print(f"\nRaw state dict with {len(checkpoint)} parameters")
            print("\nFirst 20 parameters:")
            for i, (name, tensor) in enumerate(checkpoint.items()):
                if i >= 20:
                    print(f"  ... and {len(checkpoint) - 20} more")
                    break
                if hasattr(tensor, 'shape'):
                    print(f"  {name}: {tensor.shape}")
                else:
                    print(f"  {name}: {type(tensor)}")
    
    print("\n" + "=" * 60)
    print("INSPECTION COMPLETE")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Inspect a checkpoint")
    parser.add_argument('checkpoint', type=str, help="Path to checkpoint file")
    args = parser.parse_args()
    
    inspect_checkpoint(args.checkpoint)


if __name__ == "__main__":
    main()
