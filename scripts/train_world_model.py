#!/usr/bin/env python3
"""
Complete training pipeline for the Unified World Model.

This script trains all components of the world model in phases:
1. Vision encoder (contrastive learning on images)
2. Audio encoder (contrastive learning on audio)
3. Cross-modal fusion (alignment on paired data)
4. Temporal model + Dynamics (video prediction)

Usage:
    # Train all phases
    python scripts/train_world_model.py --config configs/training_config.yaml
    
    # Train specific phase
    python scripts/train_world_model.py --config configs/training_config.yaml --phase vision
    
    # Resume training
    python scripts/train_world_model.py --config configs/training_config.yaml --resume checkpoints/latest.pth

Requirements:
    - GPU: A100 40GB (or similar)
    - Time: ~150-180 hours total
    - Storage: ~500GB for datasets
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.world_model.unified_world_model import UnifiedWorldModel, WorldModelConfig
from src.encoders.vision_encoder import VisionEncoderConfig
from src.encoders.audio_encoder import AudioEncoderConfig


def load_config(path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def setup_logging(config: Dict[str, Any]) -> Optional[Any]:
    """Setup wandb logging if enabled."""
    if config.get('logging', {}).get('use_wandb', False):
        try:
            import wandb
            wandb.init(
                project=config['logging']['wandb_project'],
                entity=config['logging'].get('wandb_entity'),
                config=config,
            )
            return wandb
        except ImportError:
            print("wandb not installed, skipping logging")
    return None


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """
    NT-Xent (Normalized Temperature-scaled Cross Entropy) loss for contrastive learning.
    
    Args:
        z1, z2: Embeddings of two augmented views [N, D]
        temperature: Temperature parameter
        
    Returns:
        Scalar loss
    """
    N = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)  # [2N, D]
    z = F.normalize(z, dim=1)
    
    # Similarity matrix
    sim = torch.mm(z, z.t()) / temperature  # [2N, 2N]
    
    # Mask out self-similarity (diagonal)
    mask = torch.eye(2 * N, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, float('-inf'))
    
    # Positive pairs: (i, i+N) and (i+N, i)
    pos_idx = torch.cat([
        torch.arange(N, 2 * N, device=z.device),
        torch.arange(N, device=z.device)
    ])
    
    # Cross entropy loss
    loss = F.cross_entropy(sim, pos_idx)
    
    return loss


def vicreg_loss(
    z1: torch.Tensor, 
    z2: torch.Tensor,
    sim_weight: float = 25.0,
    var_weight: float = 25.0,
    cov_weight: float = 1.0,
) -> torch.Tensor:
    """
    VICReg loss: Variance-Invariance-Covariance regularization.
    
    Prevents representation collapse without negative samples.
    """
    N, D = z1.shape
    
    # Invariance loss (similarity between views)
    sim_loss = F.mse_loss(z1, z2)
    
    # Variance loss (keep std above threshold)
    std1 = torch.sqrt(z1.var(dim=0) + 1e-4)
    std2 = torch.sqrt(z2.var(dim=0) + 1e-4)
    var_loss = torch.mean(F.relu(1 - std1)) + torch.mean(F.relu(1 - std2))
    
    # Covariance loss (decorrelate dimensions)
    z1_centered = z1 - z1.mean(dim=0)
    z2_centered = z2 - z2.mean(dim=0)
    
    cov1 = (z1_centered.T @ z1_centered) / (N - 1)
    cov2 = (z2_centered.T @ z2_centered) / (N - 1)
    
    # Off-diagonal covariance
    off_diag1 = cov1.flatten()[:-1].view(D - 1, D + 1)[:, 1:].flatten()
    off_diag2 = cov2.flatten()[:-1].view(D - 1, D + 1)[:, 1:].flatten()
    
    cov_loss = (off_diag1.pow(2).sum() + off_diag2.pow(2).sum()) / D
    
    return sim_weight * sim_loss + var_weight * var_loss + cov_weight * cov_loss


class SimCLRAugmentation:
    """SimCLR-style augmentation for contrastive learning."""
    
    def __init__(self, size: int = 224):
        from torchvision import transforms
        
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
        ])
    
    def __call__(self, img):
        return self.transform(img), self.transform(img)


def train_vision_encoder(
    model: UnifiedWorldModel,
    config: Dict[str, Any],
    device: torch.device,
    wandb_logger: Optional[Any] = None,
) -> None:
    """
    Phase 1: Train vision encoder with contrastive learning.
    
    Uses SimCLR-style training with NT-Xent or VICReg loss.
    """
    print("=" * 60)
    print("PHASE 1: Training Vision Encoder")
    print("=" * 60)
    
    train_config = config['training']['vision']
    data_config = config['data']
    
    # Load dataset
    try:
        from datasets import load_dataset
        
        dataset_name = data_config.get('small_vision_dataset', 'cifar100')
        print(f"Loading dataset: {dataset_name}")
        
        dataset = load_dataset(
            dataset_name,
            split="train",
            cache_dir=data_config['cache_dir'],
        )
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        print("Using random data for testing...")
        
        # Create dummy dataset for testing
        class DummyDataset:
            def __len__(self):
                return 1000
            def __getitem__(self, idx):
                return {'image': torch.randn(3, 224, 224)}
        
        dataset = DummyDataset()
    
    # Augmentation
    augment = SimCLRAugmentation(224)
    
    def collate_fn(batch):
        from PIL import Image
        import numpy as np
        
        images = []
        for item in batch:
            # Handle different data formats
            if isinstance(item, dict):
                # HuggingFace dataset format
                if 'image' in item:
                    img = item['image']
                elif 'img' in item:
                    img = item['img']
                else:
                    # Try first key that might be an image
                    for key in ['pixel_values', 'data']:
                        if key in item:
                            img = item[key]
                            break
                    else:
                        raise ValueError(f"Cannot find image in dict with keys: {item.keys()}")
            elif isinstance(item, (tuple, list)):
                img = item[0]  # Assume (image, label) format
            else:
                img = item
            
            # Convert to PIL Image if needed
            if isinstance(img, Image.Image):
                img = img.convert('RGB')
            elif isinstance(img, np.ndarray):
                if img.ndim == 2:  # Grayscale
                    img = Image.fromarray(img).convert('RGB')
                elif img.shape[0] in [1, 3]:  # CHW format
                    img = img.transpose(1, 2, 0)
                    if img.shape[2] == 1:
                        img = np.repeat(img, 3, axis=2)
                    img = Image.fromarray(img.astype(np.uint8)).convert('RGB')
                else:  # HWC format
                    img = Image.fromarray(img.astype(np.uint8)).convert('RGB')
            elif torch.is_tensor(img):
                img = img.numpy()
                if img.ndim == 2:
                    img = Image.fromarray(img).convert('RGB')
                elif img.shape[0] in [1, 3]:
                    img = img.transpose(1, 2, 0)
                    if img.shape[2] == 1:
                        img = np.repeat(img, 3, axis=2)
                    img = Image.fromarray((img * 255).astype(np.uint8)).convert('RGB')
                else:
                    img = Image.fromarray((img * 255).astype(np.uint8)).convert('RGB')
            
            images.append(img)
        
        # Apply augmentation and stack
        view1_list = []
        view2_list = []
        for img in images:
            v1, v2 = augment(img)
            view1_list.append(v1)
            view2_list.append(v2)
        
        view1 = torch.stack(view1_list)
        view2 = torch.stack(view2_list)
        return view1, view2
    
    loader = DataLoader(
        dataset,
        batch_size=train_config['batch_size'],
        shuffle=True,
        num_workers=config['training']['general']['num_workers'],
        collate_fn=collate_fn,
        pin_memory=config['training']['general']['pin_memory'],
        drop_last=True,
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.vision_encoder.parameters(),
        lr=train_config['learning_rate'],
        weight_decay=train_config['weight_decay'],
        betas=tuple(config['optimizer']['betas']),
    )
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=train_config['epochs'],
        eta_min=config['scheduler']['min_lr'],
    )
    
    # Mixed precision
    scaler = torch.amp.GradScaler('cuda') if config['training']['general']['mixed_precision'] else None
    
    # Training loop
    model.vision_encoder.train()
    global_step = 0
    
    for epoch in range(train_config['epochs']):
        epoch_loss = 0
        num_batches = 0
        
        for batch_idx, (view1, view2) in enumerate(loader):
            view1, view2 = view1.to(device), view2.to(device)
            
            with torch.amp.autocast('cuda', enabled=scaler is not None):
                # Encode both views
                z1 = model.vision_encoder(view1)
                z2 = model.vision_encoder(view2)
                
                # Pool spatial dimensions
                z1 = z1.mean(dim=(2, 3))  # [B, D]
                z2 = z2.mean(dim=(2, 3))  # [B, D]
                
                # Loss
                loss = nt_xent_loss(z1, z2, train_config.get('temperature', 0.5))
            
            # Backward
            optimizer.zero_grad()
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.vision_encoder.parameters(),
                    config['training']['general']['gradient_clip']
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.vision_encoder.parameters(),
                    config['training']['general']['gradient_clip']
                )
                optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1
            
            # Logging
            if global_step % config['training']['general']['log_every_n_steps'] == 0:
                print(f"  Step {global_step} | Loss: {loss.item():.4f}")
                if wandb_logger:
                    wandb_logger.log({
                        "vision/loss": loss.item(),
                        "vision/lr": scheduler.get_last_lr()[0],
                        "vision/epoch": epoch,
                    })
        
        scheduler.step()
        
        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch + 1}/{train_config['epochs']} | Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % config['training']['general']['save_every_n_epochs'] == 0:
            checkpoint_path = Path(data_config['checkpoint_dir']) / f"vision_encoder_epoch{epoch + 1}.pth"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.vision_encoder.state_dict(), checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")


class AudioAugment:
    """Audio augmentation for contrastive learning (two views of same clip)."""
    
    def __init__(self, sample_rate: int = 16000, max_len: int = 48000):
        self.sample_rate = sample_rate
        self.max_len = max_len
    
    def __call__(self, waveform: torch.Tensor) -> tuple:
        """Return two augmented views for contrastive learning."""
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        # Ensure consistent length
        if waveform.shape[-1] > self.max_len:
            start = random.randint(0, waveform.shape[-1] - self.max_len)
            waveform = waveform[..., start:start + self.max_len]
        elif waveform.shape[-1] < self.max_len:
            pad = self.max_len - waveform.shape[-1]
            waveform = F.pad(waveform, (0, pad))
        
        waveform = waveform.squeeze(0)
        
        # View 1: time stretch (0.8-1.2x) via resample
        stretch = random.uniform(0.9, 1.1)
        new_len = int(waveform.shape[-1] * stretch)
        view1 = F.interpolate(
            waveform.unsqueeze(0).unsqueeze(0),
            size=new_len,
            mode='linear',
            align_corners=False,
        ).squeeze()
        if view1.shape[-1] > self.max_len:
            view1 = view1[..., :self.max_len]
        elif view1.shape[-1] < self.max_len:
            view1 = F.pad(view1, (0, self.max_len - view1.shape[-1]))
        
        # View 2: add noise + volume
        view2 = waveform + torch.randn_like(waveform, device=waveform.device) * 0.005
        view2 = view2 * random.uniform(0.8, 1.2)
        
        return view1, view2


def _build_audio_dataset(config: Dict[str, Any]):
    """Build audio dataset: SpeechCommands (torchaudio) > HuggingFace > synthetic fallback."""
    sample_rate = config.get('model', {}).get('audio', {}).get('sample_rate', 16000)
    max_len = config['training']['audio'].get('max_audio_length', 48000)
    
    # 1. Try torchaudio SpeechCommands (no FFmpeg, fast)
    try:
        import torchaudio
        print("Loading SpeechCommands (torchaudio)...")
        speech_commands = torchaudio.datasets.SPEECHCOMMANDS(
            root="./data/speech_commands",
            url="speech_commands_v0.02",
            download=True,
            subset=None,
        )
        
        class SpeechCommandsWrapper:
            def __init__(self, ds, target_len, target_sr):
                self.ds = ds
                self.target_len = target_len
                self.target_sr = target_sr
            
            def __len__(self):
                return len(self.ds)
            
            def __getitem__(self, idx):
                waveform, sr, *_ = self.ds[idx]
                if sr != self.target_sr:
                    import torchaudio.transforms as T
                    resampler = T.Resample(sr, self.target_sr)
                    waveform = resampler(waveform)
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0)
                if waveform.shape[-1] > self.target_len:
                    start = random.randint(0, waveform.shape[-1] - self.target_len)
                    waveform = waveform[start:start + self.target_len]
                elif waveform.shape[-1] < self.target_len:
                    waveform = F.pad(waveform, (0, self.target_len - waveform.shape[-1]))
                return waveform.squeeze(0)
        
        dataset = SpeechCommandsWrapper(speech_commands, max_len, sample_rate)
        print(f"  Loaded {len(dataset)} audio samples (SpeechCommands)")
        return dataset, sample_rate, max_len
    except Exception as e:
        print(f"SpeechCommands failed: {e}")
    
    # 2. Try HuggingFace speech_commands
    try:
        from datasets import load_dataset
        print("Loading speech_commands (HuggingFace)...")
        ds = load_dataset(
            "speech_commands",
            "v0.02",
            split="train",
            cache_dir=config.get('data', {}).get('cache_dir', './data/huggingface_cache'),
        )
        
        class HFAudioWrapper:
            def __init__(self, ds, target_len, target_sr):
                self.ds = ds
                self.target_len = target_len
                self.target_sr = target_sr
            
            def __len__(self):
                return len(self.ds)
            
            def __getitem__(self, idx):
                item = self.ds[idx]
                waveform = torch.tensor(item['audio']['array'], dtype=torch.float32)
                sr = item['audio']['sampling_rate']
                if sr != self.target_sr:
                    waveform = F.interpolate(
                        waveform.unsqueeze(0).unsqueeze(0),
                        size=int(waveform.shape[-1] * self.target_sr / sr),
                        mode='linear',
                    ).squeeze()
                if waveform.shape[-1] > self.target_len:
                    start = random.randint(0, waveform.shape[-1] - self.target_len)
                    waveform = waveform[start:start + self.target_len]
                elif waveform.shape[-1] < self.target_len:
                    waveform = F.pad(waveform, (0, self.target_len - waveform.shape[-1]))
                return waveform
        
        dataset = HFAudioWrapper(ds, max_len, sample_rate)
        print(f"  Loaded {len(dataset)} audio samples (HuggingFace)")
        return dataset, sample_rate, max_len
    except Exception as e:
        print(f"HuggingFace speech_commands failed: {e}")
    
    # 3. Synthetic fallback
    print("Using synthetic audio fallback (random waveforms)")
    
    class SyntheticAudioDataset:
        def __init__(self, n=2000, sr=16000, length=48000):
            self.n = n
            self.sr = sr
            self.length = length
        
        def __len__(self):
            return self.n
        
        def __getitem__(self, idx):
            return torch.randn(self.length) * 0.1
    
    return SyntheticAudioDataset(2000, sample_rate, max_len), sample_rate, max_len


def train_audio_encoder(
    model: UnifiedWorldModel,
    config: Dict[str, Any],
    device: torch.device,
    wandb_logger: Optional[Any] = None,
) -> None:
    """
    Phase 2: Train audio encoder with contrastive learning (NT-Xent).
    Uses SpeechCommands when available, synthetic otherwise.
    """
    print("=" * 60)
    print("PHASE 2: Training Audio Encoder")
    print("=" * 60)
    
    train_config = config['training']['audio']
    data_config = config['data']
    
    # Build real audio dataset
    dataset, sample_rate, max_len = _build_audio_dataset(config)
    augment = AudioAugment(sample_rate, max_len)
    
    def collate_fn(batch):
        view1_list, view2_list = [], []
        for wav in batch:
            v1, v2 = augment(wav if isinstance(wav, torch.Tensor) else torch.tensor(wav, dtype=torch.float32))
            view1_list.append(v1)
            view2_list.append(v2)
        return torch.stack(view1_list), torch.stack(view2_list)
    
    loader = DataLoader(
        dataset,
        batch_size=train_config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config['training']['general']['num_workers'],
        pin_memory=config['training']['general'].get('pin_memory', False),
        drop_last=True,
    )
    
    optimizer = torch.optim.AdamW(
        model.audio_encoder.parameters(),
        lr=train_config['learning_rate'],
        weight_decay=train_config['weight_decay'],
        betas=tuple(config.get('optimizer', {}).get('betas', [0.9, 0.999])),
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=train_config['epochs'],
        eta_min=config.get('scheduler', {}).get('min_lr', 1e-6),
    )
    
    # Mixed precision (PyTorch 2.x)
    use_amp = config['training']['general'].get('mixed_precision', False)
    scaler = torch.amp.GradScaler('cuda') if (use_amp and device.type == 'cuda') else None
    
    model.audio_encoder.train()
    
    for epoch in range(train_config['epochs']):
        epoch_loss = 0
        num_batches = 0
        
        for batch_idx, (view1, view2) in enumerate(loader):
            view1, view2 = view1.to(device), view2.to(device)
            
            with torch.amp.autocast('cuda', enabled=scaler is not None):
                z1 = model.audio_encoder(view1)
                z2 = model.audio_encoder(view2)
                if z1.dim() > 2:
                    z1 = z1.mean(dim=(2, 3)) if z1.dim() == 4 else z1.mean(dim=-1)
                if z2.dim() > 2:
                    z2 = z2.mean(dim=(2, 3)) if z2.dim() == 4 else z2.mean(dim=-1)
                loss = nt_xent_loss(z1, z2, train_config.get('temperature', 0.5))
            
            if not torch.isfinite(loss):
                continue
            
            optimizer.zero_grad()
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.audio_encoder.parameters(), config['training']['general']['gradient_clip'])
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.audio_encoder.parameters(), config['training']['general']['gradient_clip'])
                optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        scheduler.step()
        avg_loss = epoch_loss / max(num_batches, 1)
        print(f"Epoch {epoch + 1}/{train_config['epochs']} | Loss: {avg_loss:.4f}")
        
        if (epoch + 1) % config['training']['general']['save_every_n_epochs'] == 0:
            ckpt_path = Path(data_config['checkpoint_dir']) / f"audio_encoder_epoch{epoch + 1}.pth"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.audio_encoder.state_dict(), ckpt_path)
            print(f"  Saved: {ckpt_path}")


def train_fusion_and_temporal(
    model: UnifiedWorldModel,
    config: Dict[str, Any],
    device: torch.device,
    wandb_logger: Optional[Any] = None,
    data_dir: Optional[str] = None,
) -> None:
    """
    Phase 3 & 4: Train fusion and temporal model on multi-modal data.
    Uses real Greatest Hits data if data_dir is provided.
    """
    print("=" * 60)
    print("PHASE 3-4: Training Fusion & Temporal Model")
    print("=" * 60)
    
    train_config = config['training']['fusion']
    epochs = train_config.get('epochs', 50)
    # Batch size from config, capped for GPU memory (8 for RTX 3050, 4 for smaller)
    batch_size = min(train_config.get('batch_size', 8), 8)
    lr = train_config.get('learning_rate', 1e-4)
    print(f"Using batch_size={batch_size} to fit in GPU memory")
    
    # Try to load real data: 1) Greatest Hits, 2) CIFAR+SpeechCommands fallback
    use_real_data = False
    
    if data_dir and Path(data_dir).exists():
        try:
            sys.path.insert(0, str(Path(__file__).parent))
            from train_multimodal import GreatestHitsDataset
            
            print(f"Loading Greatest Hits from: {data_dir}")
            train_dataset = GreatestHitsDataset(data_dir, split='train', augment=True)
            val_dataset = GreatestHitsDataset(data_dir, split='val', augment=False)
            
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True,
                num_workers=config['training']['general']['num_workers'],
                pin_memory=config['training']['general'].get('pin_memory', True),
            )
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False,
                num_workers=config['training']['general']['num_workers'],
            )
            print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
            use_real_data = True
        except Exception as e:
            print(f"Greatest Hits failed: {e}")
    
    if not use_real_data:
        try:
            from torch.utils.data import Dataset as TorchDataset
            
            class CifarSpeechDataset(TorchDataset):
                """Paired CIFAR (as video) + SpeechCommands (as audio)."""
                
                def __init__(self, split, n_frames=8, sz=224, audio_len=16000, cache_dir='./data/huggingface_cache'):
                    from datasets import load_dataset
                    self.n_frames, self.sz, self.audio_len = n_frames, sz, audio_len
                    self.img_ds = load_dataset('cifar100', split=split, cache_dir=cache_dir)
                    try:
                        import torchaudio
                        self.audio_ds = torchaudio.datasets.SPEECHCOMMANDS(
                            root='./data/speech_commands', url='speech_commands_v0.02',
                            download=True, subset=None)
                        self._use_ta = True
                    except Exception:
                        self.audio_ds = load_dataset('speech_commands', 'v0.02', split=split,
                                                     cache_dir=cache_dir)
                        self._use_ta = False
                    self.len = min(len(self.img_ds), len(self.audio_ds))
                
                def __len__(self):
                    return self.len
                
                def __getitem__(self, idx):
                    import numpy as np
                    img = self.img_ds[idx]['img']
                    if hasattr(img, 'convert'):
                        img = np.array(img.convert('RGB'))
                    else:
                        img = np.array(img)
                    if img.shape[0] in (1, 3):
                        img = np.transpose(img, (1, 2, 0))
                    img = torch.from_numpy(img).float() / 255.0
                    img = img.permute(2, 0, 1)
                    img = F.interpolate(img.unsqueeze(0), size=(self.sz, self.sz),
                                       mode='bilinear', align_corners=False).squeeze(0)
                    video = img.unsqueeze(0).expand(self.n_frames, -1, -1, -1)
                    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
                    video = ((video.permute(0, 2, 3, 1).unsqueeze(0) - mean) / std).squeeze(0).permute(0, 3, 1, 2)
                    
                    if self._use_ta:
                        wav, sr, *_ = self.audio_ds[idx]
                        if sr != 16000:
                            wav = F.interpolate(wav.unsqueeze(0).unsqueeze(0),
                                                size=int(wav.shape[-1] * 16000 / sr),
                                                mode='linear').squeeze()
                        if wav.dim() > 1:
                            wav = wav.mean(dim=0)
                    else:
                        wav = torch.tensor(self.audio_ds[idx]['audio']['array'], dtype=torch.float32)
                    
                    if wav.shape[-1] > self.audio_len:
                        wav = wav[..., :self.audio_len]
                    elif wav.shape[-1] < self.audio_len:
                        wav = F.pad(wav, (0, self.audio_len - wav.shape[-1]))
                    
                    return {'video': video, 'audio': wav.squeeze(0)}
            
            cache = config.get('data', {}).get('cache_dir', './data/huggingface_cache')
            train_dataset = CifarSpeechDataset('train', cache_dir=cache)
            val_dataset = CifarSpeechDataset('test', cache_dir=cache)
            
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True,
                num_workers=config['training']['general']['num_workers'],
                pin_memory=config['training']['general'].get('pin_memory', True),
            )
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False,
                num_workers=config['training']['general']['num_workers'],
            )
            print(f"Using CIFAR+SpeechCommands fallback: {len(train_dataset)} train, {len(val_dataset)} val")
            use_real_data = True
        except Exception as e:
            print(f"CIFAR+Speech fallback failed: {e}")
    
    if not use_real_data:
        print("WARNING: Using synthetic data. Provide --data-dir for Greatest Hits or run: python scripts/download_data.py --local-test")
    
    # Train all parameters (not just fusion) for end-to-end training
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        num_batches = 0
        
        if use_real_data:
            for batch in train_loader:
                video = batch['video'].to(device)
                audio = batch['audio'].to(device)
                
                optimizer.zero_grad()
                
                # Encode video frames - process frame by frame to save memory
                B, T, C, H, W = video.shape
                
                # Process frames in chunks to avoid OOM (no autocast, use float32)
                vision_features_list = []
                chunk_size = 4  # Process 4 frames at a time
                for t in range(0, T, chunk_size):
                    t_end = min(t + chunk_size, T)
                    frames_chunk = video[:, t:t_end].reshape(-1, C, H, W)
                    chunk_features = model.vision_encoder(frames_chunk)
                    chunk_features = chunk_features.mean(dim=(2, 3))
                    vision_features_list.append(chunk_features.view(B, t_end - t, -1).detach())
                    del frames_chunk, chunk_features
                
                vision_features = torch.cat(vision_features_list, dim=1)  # [B, T, D]
                vision_features.requires_grad_(True)
                
                # Encode audio
                audio_features = model.audio_encoder(audio)
                
                # Contrastive loss between video and audio
                video_proj = F.normalize(vision_features.mean(dim=1).float(), dim=1)
                audio_proj = F.normalize(audio_features.float(), dim=1)
                
                logits = torch.matmul(video_proj, audio_proj.T) / 0.07
                labels = torch.arange(B, device=device)
                loss = F.cross_entropy(logits, labels)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
                # Clear cache periodically
                if num_batches % 10 == 0:
                    torch.cuda.empty_cache()
        else:
            # Synthetic data fallback
            for _ in range(50):
                vision = torch.randn(batch_size, 16, 512, device=device)
                audio = torch.randn(batch_size, 512, device=device)
                
                fused = model.fusion(vision=vision, audio=audio.unsqueeze(1).expand(-1, 16, -1))
                loss = fused['fused'].pow(2).mean() if isinstance(fused, dict) else fused.pow(2).mean()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
        
        scheduler.step()
        avg_loss = epoch_loss / max(num_batches, 1)
        
        # Validation
        val_loss = 0
        if use_real_data:
            model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    video = batch['video'].to(device)
                    audio = batch['audio'].to(device)
                    
                    B, T, C, H, W = video.shape
                    
                    # Process frames in chunks
                    vision_features_list = []
                    chunk_size = 4
                    for t in range(0, T, chunk_size):
                        t_end = min(t + chunk_size, T)
                        frames_chunk = video[:, t:t_end].reshape(-1, C, H, W)
                        chunk_features = model.vision_encoder(frames_chunk)
                        chunk_features = chunk_features.mean(dim=(2, 3))
                        vision_features_list.append(chunk_features.view(B, t_end - t, -1))
                        del frames_chunk, chunk_features
                    
                    vision_features = torch.cat(vision_features_list, dim=1)
                    audio_features = model.audio_encoder(audio)
                    
                    video_proj = F.normalize(vision_features.mean(dim=1).float(), dim=1)
                    audio_proj = F.normalize(audio_features.float(), dim=1)
                    
                    logits = torch.matmul(video_proj, audio_proj.T) / 0.07
                    labels = torch.arange(B, device=device)
                    val_loss += F.cross_entropy(logits, labels).item()
                
                torch.cuda.empty_cache()
            
            val_loss /= len(val_loader)
        
        print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_loss:.4f}" + 
              (f" | Val Loss: {val_loss:.4f}" if use_real_data else ""))
        
        # Save best model
        if use_real_data and val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 
                      Path(config['data']['checkpoint_dir']) / "fusion_best.pth")
            print(f"  Saved best model (val_loss: {val_loss:.4f})")


def main():
    parser = argparse.ArgumentParser(description="Train NSCA World Model")
    parser.add_argument('--config', type=str, required=True, help="Path to config YAML")
    parser.add_argument('--phase', type=str, default='all', 
                       choices=['all', 'vision', 'audio', 'fusion', 'temporal'],
                       help="Training phase to run")
    parser.add_argument('--resume', type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument('--device', type=str, default='cuda', help="Device (cuda/cpu)")
    parser.add_argument('--data-dir', type=str, default=None, 
                       help="Path to real training data (e.g., /workspace/vis-data)")
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Set seed
    torch.manual_seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['seed'])
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Setup logging
    wandb_logger = setup_logging(config)
    
    # Create directories
    Path(config['data']['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['data']['log_dir']).mkdir(parents=True, exist_ok=True)
    
    # Build model
    print("Building model...")
    model_config = WorldModelConfig.from_dict(config['model'])
    model = UnifiedWorldModel(model_config).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Resume if specified
    if args.resume:
        print(f"Resuming from: {args.resume}")
        model.load_state_dict(torch.load(args.resume, map_location=device))
    
    # Training phases
    start_time = time.time()
    
    if args.phase in ('all', 'vision'):
        train_vision_encoder(model, config, device, wandb_logger)
    
    if args.phase in ('all', 'audio'):
        train_audio_encoder(model, config, device, wandb_logger)
    
    if args.phase in ('all', 'fusion', 'temporal'):
        train_fusion_and_temporal(model, config, device, wandb_logger, args.data_dir)
    
    # Save final model
    final_path = Path(config['data']['checkpoint_dir']) / "world_model_final.pth"
    torch.save(model.state_dict(), final_path)
    print(f"Saved final model: {final_path}")
    
    elapsed = time.time() - start_time
    print(f"\nTraining complete! Total time: {elapsed / 3600:.2f} hours")
    
    if wandb_logger:
        wandb_logger.finish()


if __name__ == "__main__":
    main()
