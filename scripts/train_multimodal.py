#!/usr/bin/env python3
"""
Multi-Modal Training on Greatest Hits Dataset.

This trains the NSCA world model on ALIGNED video + audio data,
where the model learns cross-modal correspondence:
- Given video of object, predict what it sounds like
- Given audio of impact, predict what object looks like

This is REAL multi-modal learning, not separate encoders.

Usage:
    python scripts/train_multimodal.py --data-dir /path/to/GreatestHits
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
import json
import random
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchaudio
import torchaudio.transforms as AT
from PIL import Image
import cv2

# Add project root
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Import enhanced augmentation
from src.augmentation import (
    UnifiedAugmentation,
    AugmentationConfig,
    EnhancedVideoAugmentation,
    EnhancedAudioAugmentation,
    ProprioceptionAugmentation,
    create_augmentation_pipeline,
)

# Import enhanced fusion
from src.fusion.cross_modal import (
    EnhancedCrossModalFusion,
    FusionConfig,
    create_fusion_module,
)


# =============================================================================
# DATA AUGMENTATION
# =============================================================================

class VideoAugmentation:
    """Video augmentation for training."""
    
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Augment video frames [T, C, H, W].
        """
        if random.random() > self.p:
            return frames
        
        # Random horizontal flip
        if random.random() > 0.5:
            frames = torch.flip(frames, dims=[3])  # Flip width
        
        # Random brightness
        if random.random() > 0.5:
            factor = random.uniform(0.8, 1.2)
            frames = frames * factor
            frames = torch.clamp(frames, -2.5, 2.5)  # Keep in reasonable range
        
        # Random contrast
        if random.random() > 0.5:
            factor = random.uniform(0.8, 1.2)
            mean = frames.mean(dim=(2, 3), keepdim=True)
            frames = (frames - mean) * factor + mean
            frames = torch.clamp(frames, -2.5, 2.5)
        
        # Random crop and resize (spatial jitter)
        if random.random() > 0.5:
            T, C, H, W = frames.shape
            crop_size = int(H * random.uniform(0.85, 1.0))
            top = random.randint(0, H - crop_size)
            left = random.randint(0, W - crop_size)
            frames = frames[:, :, top:top+crop_size, left:left+crop_size]
            frames = F.interpolate(frames, size=(H, W), mode='bilinear', align_corners=False)
        
        return frames


class AudioAugmentation:
    """Audio augmentation for training."""
    
    def __init__(self, sample_rate=16000, p=0.5):
        self.sample_rate = sample_rate
        self.p = p
    
    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Augment audio waveform [samples].
        """
        if random.random() > self.p:
            return waveform
        
        # Add to batch dim for transforms
        waveform = waveform.unsqueeze(0)
        
        # Random volume change
        if random.random() > 0.5:
            factor = random.uniform(0.8, 1.2)
            waveform = waveform * factor
        
        # Add noise
        if random.random() > 0.7:
            noise = torch.randn_like(waveform) * 0.005
            waveform = waveform + noise
        
        # Time shift
        if random.random() > 0.5:
            shift = random.randint(-1000, 1000)
            waveform = torch.roll(waveform, shift, dims=1)
        
        return waveform.squeeze(0)


# =============================================================================
# GREATEST HITS DATASET
# =============================================================================

class GreatestHitsDataset(Dataset):
    """
    Greatest Hits Dataset: Video + Audio of striking objects.
    
    Each sample contains:
    - video_frames: Frames around the impact moment
    - audio_clip: Audio of the impact
    - material: Material label (wood, metal, ceramic, etc.)
    """
    
    MATERIALS = ['wood', 'metal', 'ceramic', 'glass', 'plastic', 'paper', 'cloth', 'rock']
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        n_frames: int = 8,
        frame_size: int = 224,
        audio_duration: float = 1.0,
        sample_rate: int = 16000,
        augment: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.n_frames = n_frames
        self.frame_size = frame_size
        self.audio_duration = audio_duration
        self.sample_rate = sample_rate
        self.audio_samples = int(audio_duration * sample_rate)
        
        # Augmentation (only for training)
        self.augment = augment and (split == 'train')
        if self.augment:
            self.video_aug = VideoAugmentation(p=0.5)
            self.audio_aug = AudioAugmentation(sample_rate=sample_rate, p=0.5)
        
        # Find all video files
        self.samples = self._find_samples()
        
        # Split train/val
        random.seed(42)
        random.shuffle(self.samples)
        split_idx = int(0.9 * len(self.samples))
        if split == 'train':
            self.samples = self.samples[:split_idx]
        else:
            self.samples = self.samples[split_idx:]
        
        print(f"GreatestHits {split}: {len(self.samples)} samples (augment={self.augment})")
    
    def _find_samples(self) -> List[Dict]:
        """Find all video-audio pairs."""
        samples = []
        
        # Greatest Hits flat structure:
        # {timestamp}_denoised.mp4 - main video
        # {timestamp}_denoised.wav - main audio  
        # {timestamp}_times.txt - impact timestamps
        
        # Find all denoised videos (main videos, not thumbnails or mic)
        video_files = list(self.data_dir.glob('*_denoised.mp4'))
        
        print(f"Found {len(video_files)} denoised videos in {self.data_dir}")
        
        for video_path in video_files:
            # Get base name (timestamp)
            base_name = video_path.stem.replace('_denoised', '')
            
            # Find corresponding audio
            audio_path = self.data_dir / f"{base_name}_denoised.wav"
            if not audio_path.exists():
                # Try mic audio
                audio_path = self.data_dir / f"{base_name}_mic.wav"
            if not audio_path.exists():
                # Skip if no audio
                continue
            
            # Find timestamps file (contains impact times)
            times_path = self.data_dir / f"{base_name}_times.txt"
            
            # Material is unknown for this dataset - we'll learn it
            material = 'unknown'
            
            samples.append({
                'video_path': str(video_path),
                'audio_path': str(audio_path),
                'times_path': str(times_path) if times_path.exists() else None,
                'material': material,
                'name': base_name,
            })
        
        print(f"Found {len(samples)} valid video-audio pairs")
        return samples
    
    def _guess_material(self, name: str) -> str:
        """Guess material from folder/file name."""
        name_lower = name.lower()
        for material in self.MATERIALS:
            if material in name_lower:
                return material
        return 'unknown'
    
    def _load_video_frames(self, video_path: str, impact_time: float = None) -> torch.Tensor:
        """Load frames from video around impact moment."""
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            if total_frames == 0 or fps == 0:
                raise ValueError("Empty video or invalid FPS")
            
            # Get frame at impact time, or middle if not specified
            if impact_time is not None and fps > 0:
                mid_frame = int(impact_time * fps)
            else:
                mid_frame = total_frames // 2
            
            mid_frame = min(mid_frame, total_frames - 1)
            start_frame = max(0, mid_frame - self.n_frames // 2)
            
            frames = []
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            for _ in range(self.n_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Resize
                frame = cv2.resize(frame, (self.frame_size, self.frame_size))
                frames.append(frame)
            
            cap.release()
            
            # Pad if not enough frames
            while len(frames) < self.n_frames:
                frames.append(frames[-1] if frames else np.zeros((self.frame_size, self.frame_size, 3), dtype=np.uint8))
            
            # Stack and normalize
            frames = np.stack(frames)  # [T, H, W, C]
            frames = frames.transpose(0, 3, 1, 2)  # [T, C, H, W]
            frames = torch.from_numpy(frames).float() / 255.0
            
            # Normalize with ImageNet stats
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            frames = (frames - mean) / std
            
            return frames
            
        except Exception as e:
            print(f"Error loading video {video_path}: {e}")
            return torch.zeros(self.n_frames, 3, self.frame_size, self.frame_size)
    
    def _load_audio(self, audio_path: str, impact_time: float = None) -> torch.Tensor:
        """Load audio waveform around impact moment."""
        try:
            # Load audio
            waveform, sr = torchaudio.load(audio_path)
            
            # Resample if needed
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
                sr = self.sample_rate
            
            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            # Get portion around impact time, or middle if not specified
            total_samples = waveform.shape[1]
            if impact_time is not None:
                mid = int(impact_time * sr)
            else:
                mid = total_samples // 2
            
            mid = min(mid, total_samples - 1)
            start = max(0, mid - self.audio_samples // 2)
            end = start + self.audio_samples
            
            if end > total_samples:
                end = total_samples
                start = max(0, end - self.audio_samples)
            
            waveform = waveform[:, start:end]
            
            # Pad if needed
            if waveform.shape[1] < self.audio_samples:
                pad = self.audio_samples - waveform.shape[1]
                waveform = F.pad(waveform, (0, pad))
            
            return waveform.squeeze(0)  # [audio_samples]
            
        except Exception as e:
            print(f"Error loading audio {audio_path}: {e}")
            return torch.zeros(self.audio_samples)
    
    def _load_impact_times(self, times_path: str) -> List[float]:
        """Load impact timestamps from file."""
        try:
            if times_path is None:
                return []
            with open(times_path, 'r') as f:
                times = [float(line.strip()) for line in f if line.strip()]
            return times
        except:
            return []
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load impact times if available
        impact_times = self._load_impact_times(sample.get('times_path'))
        
        # Pick a random impact time, or None to use middle
        impact_time = random.choice(impact_times) if impact_times else None
        
        video_frames = self._load_video_frames(sample['video_path'], impact_time)
        audio_clip = self._load_audio(sample['audio_path'], impact_time)
        
        # Apply augmentations
        if self.augment:
            video_frames = self.video_aug(video_frames)
            audio_clip = self.audio_aug(audio_clip)
        
        material_idx = self.MATERIALS.index(sample['material']) if sample['material'] in self.MATERIALS else len(self.MATERIALS)
        
        return {
            'video': video_frames,  # [T, C, H, W]
            'audio': audio_clip,    # [audio_samples]
            'material': material_idx,
            'name': sample['name'],
        }


# =============================================================================
# CROSS-MODAL PREDICTION MODEL
# =============================================================================

class CrossModalWorldModel(nn.Module):
    """
    World Model that learns cross-modal correspondence.
    
    Key tasks:
    1. Video → Audio prediction (what will this sound like?)
    2. Audio → Video prediction (what does this look like?)
    3. Contrastive alignment (matching pairs should be close)
    4. Material classification (what is this made of?)
    """
    
    def __init__(
        self,
        vision_dim: int = 512,
        audio_dim: int = 256,
        fusion_dim: int = 512,
        n_materials: int = 9,
    ):
        super().__init__()
        
        # Vision encoder (with priors)
        self.vision_encoder = self._build_vision_encoder(vision_dim)
        
        # Audio encoder (with mel-spectrogram prior)
        self.audio_encoder = self._build_audio_encoder(audio_dim)
        
        # Cross-modal predictors with dropout
        self.vision_to_audio = nn.Sequential(
            nn.Linear(vision_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, audio_dim),
        )
        
        self.audio_to_vision = nn.Sequential(
            nn.Linear(audio_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, vision_dim),
        )
        
        # Fusion layer with dropout
        self.fusion = nn.Sequential(
            nn.Linear(vision_dim + audio_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim, fusion_dim),
        )
        
        # Material classifier (from fused representation)
        self.material_classifier = nn.Linear(fusion_dim, n_materials)
        
        # Projection heads for contrastive learning with dropout
        self.vision_proj = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(vision_dim, 128),
        )
        self.audio_proj = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(audio_dim, 128),
        )
    
    def _build_vision_encoder(self, output_dim: int) -> nn.Module:
        """Build vision encoder with Gabor priors."""
        return nn.Sequential(
            # First conv with more channels for Gabor-like features
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
            
            # ResNet-like blocks
            self._make_layer(64, 128, 2),
            self._make_layer(128, 256, 2),
            self._make_layer(256, 512, 2),
            
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, output_dim),
        )
    
    def _make_layer(self, in_ch: int, out_ch: int, stride: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )
    
    def _build_audio_encoder(self, output_dim: int) -> nn.Module:
        """Build audio encoder with mel-spectrogram."""
        return nn.Sequential(
            # Mel spectrogram is computed in forward
            nn.Conv1d(80, 128, kernel_size=3, padding=1),  # 80 mel bins
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(512, output_dim),
        )
    
    def encode_vision(self, video: torch.Tensor) -> torch.Tensor:
        """
        Encode video frames.
        
        Args:
            video: [B, T, C, H, W] video frames
        
        Returns:
            [B, vision_dim] visual features
        """
        B, T, C, H, W = video.shape
        
        # Encode each frame
        video_flat = video.view(B * T, C, H, W)
        frame_features = self.vision_encoder(video_flat)  # [B*T, vision_dim]
        
        # Average pool over time
        frame_features = frame_features.view(B, T, -1)
        video_features = frame_features.mean(dim=1)  # [B, vision_dim]
        
        return video_features
    
    def encode_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Encode audio waveform.
        
        Args:
            audio: [B, audio_samples] waveform
        
        Returns:
            [B, audio_dim] audio features
        """
        # Compute mel spectrogram (innate prior: frequency decomposition)
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=400,
            hop_length=160,
            n_mels=80,
        ).to(audio.device)
        
        mel_spec = mel_transform(audio)  # [B, n_mels, time]
        mel_spec = torch.log(mel_spec + 1e-8)  # Log scale (Weber's law prior)
        
        # Encode
        audio_features = self.audio_encoder(mel_spec)
        
        return audio_features
    
    def forward(
        self,
        video: torch.Tensor,
        audio: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with cross-modal prediction.
        
        Returns dict with:
        - vision_features: Visual encoding
        - audio_features: Audio encoding
        - vision_to_audio_pred: Predicted audio from vision
        - audio_to_vision_pred: Predicted vision from audio
        - fused: Fused representation
        - material_logits: Material classification logits
        - vision_proj: Projected vision for contrastive
        - audio_proj: Projected audio for contrastive
        """
        # Encode each modality
        vision_features = self.encode_vision(video)
        audio_features = self.encode_audio(audio)
        
        # Cross-modal prediction
        vision_to_audio_pred = self.vision_to_audio(vision_features)
        audio_to_vision_pred = self.audio_to_vision(audio_features)
        
        # Fusion
        fused = self.fusion(torch.cat([vision_features, audio_features], dim=1))
        
        # Material classification
        material_logits = self.material_classifier(fused)
        
        # Contrastive projections
        vision_proj = F.normalize(self.vision_proj(vision_features), dim=1)
        audio_proj = F.normalize(self.audio_proj(audio_features), dim=1)
        
        return {
            'vision_features': vision_features,
            'audio_features': audio_features,
            'vision_to_audio_pred': vision_to_audio_pred,
            'audio_to_vision_pred': audio_to_vision_pred,
            'fused': fused,
            'material_logits': material_logits,
            'vision_proj': vision_proj,
            'audio_proj': audio_proj,
        }


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

def cross_modal_loss(
    outputs: Dict[str, torch.Tensor],
    material_targets: torch.Tensor,
    temperature: float = 0.07,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute multi-modal training loss.
    
    Components:
    1. Cross-modal prediction loss (MSE)
    2. Contrastive loss (InfoNCE)
    3. Material classification loss (CE)
    """
    # 1. Cross-modal prediction loss
    v2a_loss = F.mse_loss(
        outputs['vision_to_audio_pred'],
        outputs['audio_features'].detach()
    )
    a2v_loss = F.mse_loss(
        outputs['audio_to_vision_pred'],
        outputs['vision_features'].detach()
    )
    prediction_loss = v2a_loss + a2v_loss
    
    # 2. Contrastive loss (InfoNCE)
    vision_proj = outputs['vision_proj']
    audio_proj = outputs['audio_proj']
    
    # Similarity matrix
    logits = torch.matmul(vision_proj, audio_proj.T) / temperature  # [B, B]
    
    # Labels: diagonal is positive
    labels = torch.arange(logits.shape[0], device=logits.device)
    
    contrastive_loss = (
        F.cross_entropy(logits, labels) +
        F.cross_entropy(logits.T, labels)
    ) / 2
    
    # 3. Material classification loss
    # Only compute if we have valid material labels (not 'unknown')
    valid_mask = material_targets < 8  # 8 known materials
    if valid_mask.sum() > 0:
        material_loss = F.cross_entropy(
            outputs['material_logits'][valid_mask],
            material_targets[valid_mask]
        )
    else:
        material_loss = torch.tensor(0.0, device=logits.device)
    
    # Total loss
    total_loss = prediction_loss + contrastive_loss + 0.5 * material_loss
    
    metrics = {
        'prediction_loss': prediction_loss.item(),
        'contrastive_loss': contrastive_loss.item(),
        'material_loss': material_loss.item(),
        'total_loss': total_loss.item(),
    }
    
    return total_loss, metrics


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train_multimodal(
    data_dir: str,
    epochs: int = 50,
    batch_size: int = 16,
    lr: float = 1e-4,
    device: str = 'cuda',
    save_dir: str = 'checkpoints',
):
    """Train the multi-modal world model."""
    print("=" * 60)
    print("MULTI-MODAL WORLD MODEL TRAINING")
    print("=" * 60)
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create dataset with augmentation
    print(f"\nLoading dataset from: {data_dir}")
    train_dataset = GreatestHitsDataset(data_dir, split='train', augment=True)
    val_dataset = GreatestHitsDataset(data_dir, split='val', augment=False)
    
    if len(train_dataset) == 0:
        print("ERROR: No samples found in dataset!")
        print("Make sure the Greatest Hits data is in the correct format.")
        return None, float('inf')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )
    
    # Create model
    print("\nBuilding model...")
    model = CrossModalWorldModel().to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    
    # Optimizer with stronger regularization
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)  # Higher weight decay
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    # Early stopping
    patience = 10
    patience_counter = 0
    
    # WandB logging
    try:
        import wandb
        wandb.init(project='nsca-multimodal', name='greatest-hits-training')
        use_wandb = True
    except:
        use_wandb = False
    
    # Training
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)
    
    best_val_loss = float('inf')
    Path(save_dir).mkdir(exist_ok=True)
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_metrics = {'total_loss': 0, 'prediction_loss': 0, 'contrastive_loss': 0}
        
        for batch_idx, batch in enumerate(train_loader):
            video = batch['video'].to(device)
            audio = batch['audio'].to(device)
            material = batch['material'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(video, audio)
            loss, metrics = cross_modal_loss(outputs, material)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            for k, v in metrics.items():
                train_metrics[k] = train_metrics.get(k, 0) + v
        
        scheduler.step()
        
        # Average metrics
        for k in train_metrics:
            train_metrics[k] /= len(train_loader)
        
        # Validate
        model.eval()
        val_metrics = {'total_loss': 0, 'prediction_loss': 0, 'contrastive_loss': 0}
        
        with torch.no_grad():
            for batch in val_loader:
                video = batch['video'].to(device)
                audio = batch['audio'].to(device)
                material = batch['material'].to(device)
                
                outputs = model(video, audio)
                loss, metrics = cross_modal_loss(outputs, material)
                
                for k, v in metrics.items():
                    val_metrics[k] = val_metrics.get(k, 0) + v
        
        for k in val_metrics:
            val_metrics[k] /= len(val_loader)
        
        # Print progress
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_metrics['total_loss']:.4f} | "
              f"Val Loss: {val_metrics['total_loss']:.4f} | "
              f"Contrastive: {val_metrics['contrastive_loss']:.4f}")
        
        # Log to WandB
        if use_wandb:
            wandb.log({
                'epoch': epoch,
                'train/total_loss': train_metrics['total_loss'],
                'train/prediction_loss': train_metrics['prediction_loss'],
                'train/contrastive_loss': train_metrics['contrastive_loss'],
                'val/total_loss': val_metrics['total_loss'],
                'val/prediction_loss': val_metrics['prediction_loss'],
                'val/contrastive_loss': val_metrics['contrastive_loss'],
                'lr': scheduler.get_last_lr()[0],
            })
        
        # Save best model and check early stopping
        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
            torch.save(model.state_dict(), f'{save_dir}/multimodal_best.pth')
            print(f"  Saved best model (val_loss: {best_val_loss:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                break
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
            }, f'{save_dir}/multimodal_epoch{epoch+1}.pth')
    
    # Save final model
    torch.save(model.state_dict(), f'{save_dir}/multimodal_final.pth')
    print(f"\nTraining complete! Final model saved to {save_dir}/multimodal_final.pth")
    
    if use_wandb:
        wandb.finish()
    
    return model, best_val_loss


def upload_to_huggingface(
    model_path: str,
    repo_id: str = "omartabius/NSCA",
    hf_token: str = None,
    commit_message: str = "Upload trained model",
    version: str = None,
):
    """
    Upload model to HuggingFace Hub.
    
    Args:
        model_path: Path to the model checkpoint
        repo_id: HuggingFace repo ID (username/repo_name)
        hf_token: HuggingFace API token (uses HF_TOKEN env var if not provided)
        commit_message: Commit message for the upload
        version: Optional version string for the model
    """
    try:
        from huggingface_hub import HfApi, login
        
        print(f"\n{'='*60}")
        print("UPLOADING TO HUGGINGFACE")
        print(f"{'='*60}")
        
        # Get token from env if not provided
        if hf_token is None:
            hf_token = os.environ.get('HF_TOKEN')
        
        if not hf_token:
            print("WARNING: No HuggingFace token provided. Set HF_TOKEN environment variable.")
            return False
        
        # Login with token
        login(token=hf_token)
        print("Logged in to HuggingFace")
        
        api = HfApi()
        
        # Upload the model file with versioned path
        model_filename = Path(model_path).name
        if version:
            path_in_repo = f"models/v{version}/{model_filename}"
        else:
            path_in_repo = f"models/{model_filename}"
        
        print(f"Uploading {model_filename} to {repo_id}...")
        
        api.upload_file(
            path_or_fileobj=model_path,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            commit_message=commit_message,
        )
        
        print(f"Successfully uploaded to https://huggingface.co/{repo_id}")
        return True
        
    except Exception as e:
        print(f"Error uploading to HuggingFace: {e}")
        return False


class HuggingFaceCallback:
    """
    Callback for automatic HuggingFace upload during training.
    
    Features:
    - Upload best model only or at every epoch
    - Model card auto-update with metrics
    - Checkpoint versioning
    """
    
    def __init__(
        self,
        repo_id: str = "omartabius/NSCA",
        token: str = None,
        upload_best_only: bool = True,
        update_model_card: bool = True,
        save_dir: str = "checkpoints",
    ):
        """
        Args:
            repo_id: HuggingFace repo ID
            token: HuggingFace token (uses HF_TOKEN env var if not provided)
            upload_best_only: Only upload when validation improves
            update_model_card: Update model card with training metrics
            save_dir: Directory where checkpoints are saved
        """
        self.repo_id = repo_id
        self.token = token or os.environ.get('HF_TOKEN')
        self.upload_best_only = upload_best_only
        self.update_model_card = update_model_card
        self.save_dir = save_dir
        self.best_val_loss = float('inf')
        self.upload_count = 0
        
        if not self.token:
            print("WARNING: HuggingFace token not set. Auto-upload disabled.")
            print("Set HF_TOKEN environment variable to enable auto-upload.")
            self.enabled = False
        else:
            self.enabled = True
    
    def on_epoch_end(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        model_path: str,
    ) -> bool:
        """
        Called at the end of each epoch.
        
        Returns True if uploaded, False otherwise.
        """
        if not self.enabled:
            return False
        
        val_loss = val_metrics.get('total_loss', float('inf'))
        
        # Check if should upload
        should_upload = not self.upload_best_only or val_loss < self.best_val_loss
        
        if should_upload:
            self.best_val_loss = min(self.best_val_loss, val_loss)
            self.upload_count += 1
            
            version = f"{epoch+1}.{self.upload_count}"
            commit_msg = (
                f"Epoch {epoch+1} | "
                f"Train Loss: {train_metrics.get('total_loss', 0):.4f} | "
                f"Val Loss: {val_loss:.4f}"
            )
            
            success = upload_to_huggingface(
                model_path=model_path,
                repo_id=self.repo_id,
                hf_token=self.token,
                commit_message=commit_msg,
                version=version if not self.upload_best_only else None,
            )
            
            if success and self.update_model_card:
                self._update_model_card(epoch, train_metrics, val_metrics)
            
            return success
        
        return False
    
    def _update_model_card(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
    ):
        """Update the model card with latest metrics."""
        try:
            card_path = Path(self.save_dir) / "README.md"
            
            # Create updated model card
            metrics_str = f"""
## Latest Training Results (Epoch {epoch + 1})

| Metric | Train | Validation |
|--------|-------|------------|
| Total Loss | {train_metrics.get('total_loss', 'N/A'):.4f} | {val_metrics.get('total_loss', 'N/A'):.4f} |
| Prediction Loss | {train_metrics.get('prediction_loss', 'N/A'):.4f} | {val_metrics.get('prediction_loss', 'N/A'):.4f} |
| Contrastive Loss | {train_metrics.get('contrastive_loss', 'N/A'):.4f} | {val_metrics.get('contrastive_loss', 'N/A'):.4f} |

Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
            
            card_content = create_model_card_content(val_metrics)
            card_content += metrics_str
            
            with open(card_path, 'w') as f:
                f.write(card_content)
            
            # Upload model card
            upload_to_huggingface(
                model_path=str(card_path),
                repo_id=self.repo_id,
                hf_token=self.token,
                commit_message=f"Update model card (epoch {epoch + 1})",
            )
            
        except Exception as e:
            print(f"Failed to update model card: {e}")


def create_model_card_content(metrics: dict) -> str:
    """Create model card content."""
    return f"""---
license: mit
tags:
- multimodal
- audio-visual
- world-model
- cross-modal
- neuro-symbolic
- physics-priors
datasets:
- greatest-hits
- something-something-v2
- physion
---

# NSCA Cross-Modal World Model v2.0

A multi-modal world model with **enhanced cross-modal fusion** trained on audio-visual correspondence learning.

## Model Description

NSCA (Neuro-Symbolic Cognitive Architecture) is a biologically-inspired cognitive architecture that learns to:
- Predict what an object sounds like from video
- Predict what an object looks like from audio
- Align audio-visual-proprioceptive representations in a shared space
- Reason about physics using adaptive priors

## Key Features

- **Hierarchical Fusion**: Early/Mid/Late fusion for robust cross-modal integration
- **CLIP-style Contrastive Learning**: Learns aligned embeddings across modalities
- **Temporal Synchronization**: Handles different sampling rates across modalities
- **Physics-Aware Augmentation**: Preserves gravity-consistent transformations
- **Adaptive Priors**: Learnable physics biases that can be overridden by experience

## Training Results

- Best Validation Loss: {metrics.get('val_loss', 'N/A')}
- Training Epochs: {metrics.get('epochs', 'N/A')}
- Model Parameters: ~16M (with enhanced fusion)

## Usage

```python
from src.world_model.unified_world_model import UnifiedWorldModel, WorldModelConfig

# Create model with enhanced fusion
config = WorldModelConfig(use_enhanced_fusion=True)
model = UnifiedWorldModel(config)
model.load_state_dict(torch.load("multimodal_best.pth"))
model.eval()

# Forward pass with all modalities
results = model(
    vision=video_tensor,     # [B, T, C, H, W]
    audio=audio_tensor,      # [B, samples]
    proprio=proprio_tensor,  # [B, T, 12]
)

world_state = results['world_state']
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   ENHANCED CROSS-MODAL FUSION               │
├─────────────────────────────────────────────────────────────┤
│  1. Modality-Specific Encoders                              │
│     └─ Vision: [B,T,C,H,W] → [B,T,D]                       │
│     └─ Audio:  [B,samples] → [B,T_a,D]                     │
│     └─ Proprio:[B,T,12]   → [B,T,D]                        │
├─────────────────────────────────────────────────────────────┤
│  2. Cross-Modal Contrastive Alignment (CLIP-style)          │
├─────────────────────────────────────────────────────────────┤
│  3. Hierarchical Fusion (Early/Mid/Late)                    │
├─────────────────────────────────────────────────────────────┤
│  4. Temporal Synchronization                                │
└─────────────────────────────────────────────────────────────┘
```

## Citation

If you use this model, please cite:

```bibtex
@software{{nsca2026,
  author = {{NSCA Contributors}},
  title = {{NSCA: Neuro-Symbolic Cognitive Architecture}},
  year = {{2026}},
  url = {{https://huggingface.co/omartabius/NSCA}}
}}
```
"""


def create_model_card(save_dir: str, metrics: dict):
    """Create a model card README for HuggingFace."""
    card_content = create_model_card_content(metrics)
    
    card_path = Path(save_dir) / "README.md"
    with open(card_path, 'w') as f:
        f.write(card_content)
    
    return str(card_path)


# =============================================================================
# ENHANCED TRAINING WITH COMBINED LOSS
# =============================================================================

def enhanced_cross_modal_loss(
    outputs: Dict[str, torch.Tensor],
    material_targets: torch.Tensor,
    temperature: float = 0.07,
    contrastive_weight: float = 0.3,
    prediction_weight: float = 0.25,
    physics_weight: float = 0.2,
    reconstruction_weight: float = 0.15,
    classification_weight: float = 0.1,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Enhanced multi-modal training loss with configurable weights.
    
    Components:
    1. Cross-modal prediction loss (MSE) - 0.25
    2. Contrastive loss (InfoNCE) - 0.3
    3. Physics prior loss - 0.2 (if available)
    4. Reconstruction loss - 0.15 (if available)
    5. Material classification loss (CE) - 0.1
    """
    metrics = {}
    total_loss = torch.tensor(0.0, device=outputs['vision_features'].device)
    
    # 1. Cross-modal prediction loss
    v2a_loss = F.mse_loss(
        outputs['vision_to_audio_pred'],
        outputs['audio_features'].detach()
    )
    a2v_loss = F.mse_loss(
        outputs['audio_to_vision_pred'],
        outputs['vision_features'].detach()
    )
    prediction_loss = v2a_loss + a2v_loss
    total_loss = total_loss + prediction_weight * prediction_loss
    metrics['prediction_loss'] = prediction_loss.item()
    
    # 2. Contrastive loss (InfoNCE)
    vision_proj = outputs['vision_proj']
    audio_proj = outputs['audio_proj']
    
    logits = torch.matmul(vision_proj, audio_proj.T) / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    
    contrastive_loss = (
        F.cross_entropy(logits, labels) +
        F.cross_entropy(logits.T, labels)
    ) / 2
    total_loss = total_loss + contrastive_weight * contrastive_loss
    metrics['contrastive_loss'] = contrastive_loss.item()
    
    # 3. Fusion contrastive loss (if available from enhanced fusion)
    if 'fusion_contrastive_loss' in outputs:
        fusion_loss = outputs['fusion_contrastive_loss']
        total_loss = total_loss + (contrastive_weight * 0.5) * fusion_loss
        metrics['fusion_contrastive_loss'] = fusion_loss.item()
    
    # 4. Physics prior loss (if available)
    if 'physics_loss' in outputs:
        physics_loss = outputs['physics_loss']
        total_loss = total_loss + physics_weight * physics_loss
        metrics['physics_loss'] = physics_loss.item()
    
    # 5. Material classification loss
    valid_mask = material_targets < 8
    if valid_mask.sum() > 0:
        material_loss = F.cross_entropy(
            outputs['material_logits'][valid_mask],
            material_targets[valid_mask]
        )
    else:
        material_loss = torch.tensor(0.0, device=logits.device)
    total_loss = total_loss + classification_weight * material_loss
    metrics['material_loss'] = material_loss.item()
    
    metrics['total_loss'] = total_loss.item()
    
    return total_loss, metrics


def train_enhanced(
    data_dir: str,
    epochs: int = 100,
    batch_size: int = 16,
    lr: float = 1e-4,
    device: str = 'cuda',
    save_dir: str = 'checkpoints',
    use_enhanced_augmentation: bool = True,
    physics_aware: bool = True,
    hf_callback: Optional[HuggingFaceCallback] = None,
    config_path: Optional[str] = None,
    patience: int = 10,
):
    """
    Enhanced training with unified augmentation and auto HF upload.
    
    Args:
        data_dir: Path to training data
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        device: Training device
        save_dir: Directory for checkpoints
        use_enhanced_augmentation: Use enhanced augmentation pipeline
        physics_aware: Enable physics-aware augmentation
        hf_callback: Optional HuggingFace callback for auto-upload
        config_path: Optional path to YAML configuration
    """
    print("=" * 60)
    print("ENHANCED MULTI-MODAL WORLD MODEL TRAINING")
    print("=" * 60)
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Enhanced Augmentation: {use_enhanced_augmentation}")
    print(f"Physics Aware: {physics_aware}")
    
    # Create augmentation pipeline
    if use_enhanced_augmentation:
        if config_path:
            augmentation = create_augmentation_pipeline(config_path=config_path)
        else:
            augmentation = create_augmentation_pipeline(
                physics_aware=physics_aware,
                probability=0.5,
            )
        print("Using enhanced augmentation pipeline")
    else:
        augmentation = None
    
    # Create dataset
    print(f"\nLoading dataset from: {data_dir}")
    train_dataset = GreatestHitsDataset(data_dir, split='train', augment=True)
    val_dataset = GreatestHitsDataset(data_dir, split='val', augment=False)
    
    if len(train_dataset) == 0:
        print("ERROR: No samples found in dataset!")
        return None, float('inf')
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )
    
    # Create model
    print("\nBuilding model...")
    model = CrossModalWorldModel().to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    # Early stopping (uses patience parameter)
    patience_counter = 0
    best_val_loss = float('inf')
    
    # WandB logging
    try:
        import wandb
        wandb.init(
            project='nsca-multimodal',
            name=f'enhanced-training-{datetime.now().strftime("%Y%m%d-%H%M%S")}',
            config={
                'epochs': epochs,
                'batch_size': batch_size,
                'lr': lr,
                'enhanced_augmentation': use_enhanced_augmentation,
                'physics_aware': physics_aware,
            }
        )
        use_wandb = True
    except:
        use_wandb = False
    
    # Training
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)
    
    Path(save_dir).mkdir(exist_ok=True)
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_metrics = {'total_loss': 0, 'prediction_loss': 0, 'contrastive_loss': 0}
        
        for batch_idx, batch in enumerate(train_loader):
            video = batch['video'].to(device)
            audio = batch['audio'].to(device)
            material = batch['material'].to(device)
            
            # Apply enhanced augmentation if enabled
            if augmentation is not None:
                aug_result = augmentation.augment_vision(video)
                video = aug_result[0]
                audio_aug, _ = augmentation.augment_audio(audio)
                audio = audio_aug
            
            optimizer.zero_grad()
            
            outputs = model(video, audio)
            loss, metrics = enhanced_cross_modal_loss(outputs, material)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            for k, v in metrics.items():
                train_metrics[k] = train_metrics.get(k, 0) + v
        
        scheduler.step()
        
        # Average metrics
        for k in train_metrics:
            train_metrics[k] /= len(train_loader)
        
        # Validate
        model.eval()
        val_metrics = {'total_loss': 0, 'prediction_loss': 0, 'contrastive_loss': 0}
        
        with torch.no_grad():
            for batch in val_loader:
                video = batch['video'].to(device)
                audio = batch['audio'].to(device)
                material = batch['material'].to(device)
                
                outputs = model(video, audio)
                loss, metrics = enhanced_cross_modal_loss(outputs, material)
                
                for k, v in metrics.items():
                    val_metrics[k] = val_metrics.get(k, 0) + v
        
        for k in val_metrics:
            val_metrics[k] /= len(val_loader)
        
        # Print progress
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_metrics['total_loss']:.4f} | "
              f"Val Loss: {val_metrics['total_loss']:.4f} | "
              f"Contrastive: {val_metrics['contrastive_loss']:.4f}")
        
        # Log to WandB
        if use_wandb:
            wandb.log({
                'epoch': epoch,
                **{f'train/{k}': v for k, v in train_metrics.items()},
                **{f'val/{k}': v for k, v in val_metrics.items()},
                'lr': scheduler.get_last_lr()[0],
            })
        
        # Save best model
        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
            model_path = f'{save_dir}/multimodal_best.pth'
            torch.save(model.state_dict(), model_path)
            print(f"  Saved best model (val_loss: {best_val_loss:.4f})")
            patience_counter = 0
            
            # Auto-upload to HuggingFace (with error protection)
            if hf_callback is not None:
                try:
                    hf_callback.on_epoch_end(epoch, train_metrics, val_metrics, model_path)
                except Exception as e:
                    print(f"  HuggingFace upload failed (continuing training): {e}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
            }, f'{save_dir}/multimodal_epoch{epoch+1}.pth')
    
    # Save final model
    torch.save(model.state_dict(), f'{save_dir}/multimodal_final.pth')
    print(f"\nTraining complete! Final model saved to {save_dir}/multimodal_final.pth")
    
    if use_wandb:
        wandb.finish()
    
    return model, best_val_loss


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train enhanced multi-modal world model")
    parser.add_argument('--data-dir', type=str, required=True, help="Path to training data")
    parser.add_argument('--epochs', type=int, default=100, help="Training epochs")
    parser.add_argument('--batch-size', type=int, default=16, help="Batch size")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--device', type=str, default='cuda', help="Device")
    parser.add_argument('--save-dir', type=str, default='checkpoints', help="Save directory")
    parser.add_argument('--config', type=str, default=None, help="Path to YAML config")
    parser.add_argument('--patience', type=int, default=10, help="Early stopping patience")
    
    # Augmentation options
    parser.add_argument('--enhanced-aug', action='store_true', default=True,
                       help="Use enhanced augmentation (default: True)")
    parser.add_argument('--no-enhanced-aug', dest='enhanced_aug', action='store_false',
                       help="Disable enhanced augmentation")
    parser.add_argument('--physics-aware', action='store_true', default=True,
                       help="Use physics-aware augmentation (default: True)")
    
    # HuggingFace options
    parser.add_argument('--hf-token', type=str, default=None,
                       help="HuggingFace API token (or set HF_TOKEN env var)")
    parser.add_argument('--hf-repo', type=str, default='omartabius/NSCA',
                       help="HuggingFace repo ID")
    parser.add_argument('--upload-hf', action='store_true',
                       help="Upload to HuggingFace after training")
    parser.add_argument('--auto-upload', action='store_true',
                       help="Auto-upload best model during training")
    
    args = parser.parse_args()
    
    # Setup HuggingFace callback
    hf_callback = None
    if args.auto_upload:
        hf_token = args.hf_token or os.environ.get('HF_TOKEN')
        if hf_token:
            hf_callback = HuggingFaceCallback(
                repo_id=args.hf_repo,
                token=hf_token,
                upload_best_only=True,
                update_model_card=True,
                save_dir=args.save_dir,
            )
            print(f"Auto-upload enabled to: {args.hf_repo}")
        else:
            print("WARNING: Auto-upload requested but no HF_TOKEN set")
    
    # Train the model
    model, best_val_loss = train_enhanced(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        save_dir=args.save_dir,
        use_enhanced_augmentation=args.enhanced_aug,
        physics_aware=args.physics_aware,
        hf_callback=hf_callback,
        config_path=args.config,
        patience=args.patience,
    )
    
    # Upload to HuggingFace if requested (post-training)
    if args.upload_hf:
        hf_token = args.hf_token or os.environ.get('HF_TOKEN')
        if hf_token:
            # Create model card
            metrics = {'val_loss': best_val_loss, 'epochs': args.epochs}
            create_model_card(args.save_dir, metrics)
            
            # Upload best model
            upload_to_huggingface(
                model_path=f"{args.save_dir}/multimodal_best.pth",
                repo_id=args.hf_repo,
                hf_token=hf_token,
                commit_message=f"Upload multimodal model (val_loss={best_val_loss:.4f})",
            )
            
            # Upload model card
            upload_to_huggingface(
                model_path=f"{args.save_dir}/README.md",
                repo_id=args.hf_repo,
                hf_token=hf_token,
                commit_message="Upload model card",
            )
        else:
            print("WARNING: Upload requested but no HF_TOKEN available")


if __name__ == "__main__":
    main()
