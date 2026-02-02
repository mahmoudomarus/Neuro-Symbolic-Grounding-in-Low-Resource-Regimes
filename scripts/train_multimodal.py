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
from typing import Dict, List, Tuple, Optional
import json
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
from PIL import Image
import cv2

# Add project root
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


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
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.n_frames = n_frames
        self.frame_size = frame_size
        self.audio_duration = audio_duration
        self.sample_rate = sample_rate
        self.audio_samples = int(audio_duration * sample_rate)
        
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
        
        print(f"GreatestHits {split}: {len(self.samples)} samples")
    
    def _find_samples(self) -> List[Dict]:
        """Find all video-audio pairs."""
        samples = []
        
        # Greatest Hits structure: vis_data/video_name/
        vis_dir = self.data_dir / 'vis'
        if not vis_dir.exists():
            vis_dir = self.data_dir  # Try root
        
        for video_dir in vis_dir.iterdir():
            if not video_dir.is_dir():
                continue
            
            # Find video file
            video_files = list(video_dir.glob('*.mp4')) + list(video_dir.glob('*.avi'))
            if not video_files:
                continue
            video_path = video_files[0]
            
            # Find audio file
            audio_files = list(video_dir.glob('*.wav')) + list(video_dir.glob('*.mp3'))
            if not audio_files:
                # Try to extract from video
                audio_path = video_path  # Will extract audio from video
            else:
                audio_path = audio_files[0]
            
            # Try to get material from folder name or metadata
            material = self._guess_material(video_dir.name)
            
            samples.append({
                'video_path': str(video_path),
                'audio_path': str(audio_path),
                'material': material,
                'name': video_dir.name,
            })
        
        return samples
    
    def _guess_material(self, name: str) -> str:
        """Guess material from folder/file name."""
        name_lower = name.lower()
        for material in self.MATERIALS:
            if material in name_lower:
                return material
        return 'unknown'
    
    def _load_video_frames(self, video_path: str) -> torch.Tensor:
        """Load frames from video around middle (impact moment)."""
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            if total_frames == 0:
                raise ValueError("Empty video")
            
            # Get frames around the middle (impact usually happens there)
            mid_frame = total_frames // 2
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
    
    def _load_audio(self, audio_path: str) -> torch.Tensor:
        """Load audio waveform."""
        try:
            # Check if it's a video file (extract audio)
            if audio_path.endswith(('.mp4', '.avi')):
                # Use torchaudio to load from video
                waveform, sr = torchaudio.load(audio_path)
            else:
                waveform, sr = torchaudio.load(audio_path)
            
            # Resample if needed
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            
            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            # Get middle portion (around impact)
            total_samples = waveform.shape[1]
            mid = total_samples // 2
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
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        video_frames = self._load_video_frames(sample['video_path'])
        audio_clip = self._load_audio(sample['audio_path'])
        
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
        
        # Cross-modal predictors
        self.vision_to_audio = nn.Sequential(
            nn.Linear(vision_dim, 256),
            nn.ReLU(),
            nn.Linear(256, audio_dim),
        )
        
        self.audio_to_vision = nn.Sequential(
            nn.Linear(audio_dim, 256),
            nn.ReLU(),
            nn.Linear(256, vision_dim),
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(vision_dim + audio_dim, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, fusion_dim),
        )
        
        # Material classifier (from fused representation)
        self.material_classifier = nn.Linear(fusion_dim, n_materials)
        
        # Projection heads for contrastive learning
        self.vision_proj = nn.Linear(vision_dim, 128)
        self.audio_proj = nn.Linear(audio_dim, 128)
    
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
    
    # Create dataset
    print(f"\nLoading dataset from: {data_dir}")
    train_dataset = GreatestHitsDataset(data_dir, split='train')
    val_dataset = GreatestHitsDataset(data_dir, split='val')
    
    if len(train_dataset) == 0:
        print("ERROR: No samples found in dataset!")
        print("Make sure the Greatest Hits data is in the correct format.")
        return
    
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
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
        
        # Save best model
        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
            torch.save(model.state_dict(), f'{save_dir}/multimodal_best.pth')
            print(f"  Saved best model (val_loss: {best_val_loss:.4f})")
        
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


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train multi-modal world model")
    parser.add_argument('--data-dir', type=str, required=True, help="Path to Greatest Hits data")
    parser.add_argument('--epochs', type=int, default=50, help="Training epochs")
    parser.add_argument('--batch-size', type=int, default=16, help="Batch size")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--device', type=str, default='cuda', help="Device")
    parser.add_argument('--save-dir', type=str, default='checkpoints', help="Save directory")
    args = parser.parse_args()
    
    train_multimodal(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()
