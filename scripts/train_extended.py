#!/usr/bin/env python3
"""
Extended training script for longer convergence.

This trains the full pipeline with:
- More epochs (200)
- Learning rate scheduling
- Better checkpointing
- Resume from existing checkpoint
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from datetime import datetime
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
import yaml

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.world_model.unified_world_model import UnifiedWorldModel, WorldModelConfig
from train_multimodal import GreatestHitsDataset

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def train_extended(
    model: UnifiedWorldModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    config: dict,
    epochs: int = 200,
    lr: float = 1e-4,
    resume_checkpoint: str = None,
    use_wandb: bool = True,
):
    """
    Extended training with better optimization.
    """
    print("\n" + "=" * 60)
    print("EXTENDED TRAINING")
    print(f"Epochs: {epochs}, LR: {lr}")
    print("=" * 60)
    
    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=0.01,
        betas=(0.9, 0.999)
    )
    
    # Cosine annealing scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    
    # Resume from checkpoint if provided
    start_epoch = 0
    best_val_loss = float('inf')
    
    if resume_checkpoint and Path(resume_checkpoint).exists():
        print(f"Resuming from: {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint.get('epoch', 0)
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            print(f"  Resumed from epoch {start_epoch}, best_val_loss: {best_val_loss:.4f}")
        else:
            model.load_state_dict(checkpoint)
            print("  Loaded model weights only")
    
    # WandB logging
    if use_wandb and HAS_WANDB:
        wandb.init(
            project="nsca-extended-training",
            config={
                "epochs": epochs,
                "lr": lr,
                "batch_size": train_loader.batch_size,
                "model_params": sum(p.numel() for p in model.parameters()),
            }
        )
    
    # Training loop
    patience = 30
    no_improve = 0
    
    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0
        num_batches = 0
        
        for batch in train_loader:
            video = batch['video'].to(device)
            audio = batch['audio'].to(device)
            
            optimizer.zero_grad()
            
            B, T, C, H, W = video.shape
            
            # Process video in chunks to save memory
            chunk_size = 4
            vision_features_list = []
            
            for t in range(0, T, chunk_size):
                t_end = min(t + chunk_size, T)
                frames_chunk = video[:, t:t_end].reshape(-1, C, H, W)
                chunk_features = model.vision_encoder(frames_chunk)
                chunk_features = chunk_features.mean(dim=(2, 3))
                vision_features_list.append(chunk_features.view(B, t_end - t, -1).detach())
                del frames_chunk, chunk_features
            
            vision_features = torch.cat(vision_features_list, dim=1)
            vision_features.requires_grad_(True)
            
            # Encode audio
            audio_features = model.audio_encoder(audio)
            
            # Contrastive loss
            video_proj = F.normalize(vision_features.mean(dim=1).float(), dim=1)
            audio_proj = F.normalize(audio_features.float(), dim=1)
            
            logits = torch.matmul(video_proj, audio_proj.T) / 0.07
            labels = torch.arange(B, device=device)
            
            # Symmetric loss
            loss_v2a = F.cross_entropy(logits, labels)
            loss_a2v = F.cross_entropy(logits.T, labels)
            loss = (loss_v2a + loss_a2v) / 2
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            if num_batches % 20 == 0:
                torch.cuda.empty_cache()
        
        # Step scheduler
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                video = batch['video'].to(device)
                audio = batch['audio'].to(device)
                
                B, T, C, H, W = video.shape
                
                vision_features_list = []
                chunk_size = 4
                for t in range(0, T, chunk_size):
                    t_end = min(t + chunk_size, T)
                    frames_chunk = video[:, t:t_end].reshape(-1, C, H, W)
                    chunk_features = model.vision_encoder(frames_chunk)
                    chunk_features = chunk_features.mean(dim=(2, 3))
                    vision_features_list.append(chunk_features.view(B, t_end - t, -1))
                
                vision_features = torch.cat(vision_features_list, dim=1)
                audio_features = model.audio_encoder(audio)
                
                video_proj = F.normalize(vision_features.mean(dim=1).float(), dim=1)
                audio_proj = F.normalize(audio_features.float(), dim=1)
                
                logits = torch.matmul(video_proj, audio_proj.T) / 0.07
                labels = torch.arange(B, device=device)
                
                loss_v2a = F.cross_entropy(logits, labels)
                loss_a2v = F.cross_entropy(logits.T, labels)
                val_loss += (loss_v2a + loss_a2v).item() / 2
        
        val_loss /= len(val_loader)
        train_loss = epoch_loss / num_batches
        current_lr = scheduler.get_last_lr()[0]
        
        print(f"Epoch {epoch+1}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | LR: {current_lr:.2e}")
        
        # Logging
        if use_wandb and HAS_WANDB:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": current_lr,
            })
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
            }, 'checkpoints/extended_best.pth')
            print(f"  Saved best model (val_loss: {val_loss:.4f})")
        else:
            no_improve += 1
        
        # Periodic checkpoint
        if (epoch + 1) % 20 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
            }, f'checkpoints/extended_epoch{epoch+1}.pth')
        
        # Early stopping
        if no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break
        
        torch.cuda.empty_cache()
    
    # Save final model
    torch.save(model.state_dict(), 'checkpoints/extended_final.pth')
    print(f"\nTraining complete! Best val_loss: {best_val_loss:.4f}")
    
    if use_wandb and HAS_WANDB:
        wandb.finish()
    
    return best_val_loss


def main():
    parser = argparse.ArgumentParser(description='Extended training')
    parser.add_argument('--config', type=str, default='configs/training_config.yaml')
    parser.add_argument('--data-dir', type=str, required=True, help='Path to GreatestHits data')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--no-wandb', action='store_true', help='Disable wandb logging')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Build model
    print("\nBuilding model...")
    model_config = WorldModelConfig.from_dict(config['model'])
    model = UnifiedWorldModel(model_config).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Load data
    print(f"\nLoading data from: {args.data_dir}")
    
    train_dataset = GreatestHitsDataset(
        data_dir=args.data_dir,
        split='train',
        n_frames=8,
        audio_duration=1.0,
        augment=True
    )
    
    val_dataset = GreatestHitsDataset(
        data_dir=args.data_dir,
        split='val',
        n_frames=8,
        audio_duration=1.0,
        augment=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Train
    train_extended(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=config,
        epochs=args.epochs,
        lr=args.lr,
        resume_checkpoint=args.resume,
        use_wandb=not args.no_wandb
    )


if __name__ == '__main__':
    main()
