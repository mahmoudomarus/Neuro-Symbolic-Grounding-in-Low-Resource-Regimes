 """
NSCA Physion Benchmark Experiment
=================================
Kaggle Notebook for testing physics priors on video prediction tasks.

This notebook tests the NSCA hypothesis:
"Physics priors improve sample efficiency on DYNAMICS PREDICTION tasks"

Upload to Kaggle and enable GPU for best performance.

Author: NSCA Team
Date: 2026-01-31
"""

# %% [markdown]
# # NSCA Physics Prior Evaluation on Physion
# 
# This notebook evaluates whether physics priors (gravity, support) improve 
# sample efficiency on video prediction tasks from the Physion benchmark.
# 
# **Hypothesis**: Priors help more on DYNAMICS prediction than static classification.

# %% [code]
# Install dependencies
# !pip install torch torchvision numpy matplotlib tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from typing import Tuple, Dict, List

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %% [markdown]
# ## 1. Synthetic Physion-like Dataset
# 
# We create a synthetic dataset that mimics the Physion benchmark:
# - Video sequences of objects in physical scenarios
# - Task: Predict whether a configuration will remain stable

# %% [code]
class SyntheticPhysionDataset:
    """
    Generate synthetic video sequences for stability prediction.
    Each sample is a sequence of frames showing objects + support.
    Label = 1 if configuration is stable, 0 otherwise.
    """
    
    def __init__(self, n_samples: int, n_frames: int = 8, img_size: int = 64, seed: int = 42):
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.n_samples = n_samples
        self.n_frames = n_frames
        self.img_size = img_size
        
        self.videos, self.labels, self.metadata = self._generate()
    
    def _generate(self) -> Tuple[torch.Tensor, torch.Tensor, List[Dict]]:
        videos = []
        labels = []
        metadata = []
        
        for i in range(self.n_samples):
            video, label, meta = self._generate_sample()
            videos.append(video)
            labels.append(label)
            metadata.append(meta)
        
        return torch.stack(videos), torch.tensor(labels).float(), metadata
    
    def _generate_sample(self) -> Tuple[torch.Tensor, float, Dict]:
        """Generate a single video sequence with physics."""
        frames = []
        
        # Object properties
        obj_x = np.random.randint(15, self.img_size - 25)
        obj_y_start = np.random.randint(10, 30)
        obj_width = np.random.randint(8, 16)
        obj_height = np.random.randint(6, 12)
        obj_color = torch.rand(3) * 0.5 + 0.5
        
        # Support (table) properties
        table_x = np.random.randint(10, self.img_size - 30)
        table_width = np.random.randint(15, 25)
        table_y = self.img_size - 12
        
        # Physics simulation
        obj_center_x = obj_x + obj_width // 2
        supported = (table_x <= obj_center_x <= table_x + table_width)
        
        # Simulate frames with gravity
        obj_y = obj_y_start
        velocity = 0.0
        gravity = 2.0
        
        for f in range(self.n_frames):
            frame = torch.zeros(3, self.img_size, self.img_size)
            
            # Draw table (brown)
            frame[0, table_y:table_y+5, table_x:table_x+table_width] = 0.6
            frame[1, table_y:table_y+5, table_x:table_x+table_width] = 0.4
            frame[2, table_y:table_y+5, table_x:table_x+table_width] = 0.2
            
            # Draw object at current position
            obj_y_int = int(np.clip(obj_y, 0, self.img_size - obj_height - 1))
            frame[:, obj_y_int:obj_y_int+obj_height, obj_x:obj_x+obj_width] = obj_color.view(3, 1, 1)
            
            frames.append(frame)
            
            # Physics update
            if supported and obj_y + obj_height >= table_y:
                # Resting on table
                obj_y = table_y - obj_height
                velocity = 0
            else:
                # Falling
                velocity += gravity
                obj_y += velocity
        
        video = torch.stack(frames)  # (T, C, H, W)
        
        # Label: stable if object ends up resting on table
        final_y = obj_y
        is_stable = supported and (final_y + obj_height >= table_y - 5)
        
        meta = {
            'obj_x': obj_x, 'obj_center': obj_center_x,
            'table_x': table_x, 'table_width': table_width,
            'supported': supported, 'final_y': final_y
        }
        
        return video, 1.0 if is_stable else 0.0, meta
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return self.videos[idx], self.labels[idx]


# %% [code]
# Create datasets
print("Generating synthetic Physion-like datasets...")
train_dataset = SyntheticPhysionDataset(n_samples=500, seed=42)
test_dataset = SyntheticPhysionDataset(n_samples=200, seed=999)

print(f"Train: {len(train_dataset)} samples, {train_dataset.labels.mean().item():.1%} stable")
print(f"Test: {len(test_dataset)} samples, {test_dataset.labels.mean().item():.1%} stable")

# Visualize a sample
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
sample_video, sample_label = train_dataset[0]
for i, ax in enumerate(axes.flat):
    ax.imshow(sample_video[i].permute(1, 2, 0).numpy())
    ax.set_title(f"Frame {i}")
    ax.axis('off')
plt.suptitle(f"Sample Video (Stable: {bool(sample_label)})")
plt.tight_layout()
plt.savefig("sample_video.png")
plt.show()

# %% [markdown]
# ## 2. Models: With and Without Physics Priors
# 
# We compare two models:
# 1. **Baseline**: Pure neural network video encoder
# 2. **NSCA**: Same encoder + physics priors (gravity, support detection)

# %% [code]
class VideoEncoder(nn.Module):
    """Encode video sequence to features."""
    
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.temporal_encoder = nn.GRU(128 * 16, hidden_dim, batch_first=True)
        self.hidden_dim = hidden_dim
    
    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video: (B, T, C, H, W)
        Returns:
            features: (B, hidden_dim)
        """
        B, T, C, H, W = video.shape
        
        # Encode each frame
        frames_flat = video.view(B * T, C, H, W)
        spatial_features = self.spatial_encoder(frames_flat)
        spatial_features = spatial_features.view(B, T, -1)
        
        # Temporal encoding
        _, hidden = self.temporal_encoder(spatial_features)
        return hidden.squeeze(0)


class GravityPrior(nn.Module):
    """
    Physics prior: Objects fall unless supported.
    Estimates stability from video dynamics.
    """
    
    def __init__(self):
        super().__init__()
        self.min_weight = 0.3
        self._weight_raw = nn.Parameter(torch.tensor(0.5))
    
    @property
    def prior_weight(self):
        return self.min_weight + F.softplus(self._weight_raw)
    
    def forward(self, video: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Analyze video for physics-based stability cues.
        
        Returns:
            stability_score: (B, 1) probability of stability
            diagnostics: dict with intermediate values
        """
        B, T, C, H, W = video.shape
        
        # Detect object motion (compare first and last frames)
        first_frame = video[:, 0]
        last_frame = video[:, -1]
        
        # Motion magnitude in lower half (where falling would occur)
        lower_half = slice(H // 2, H)
        motion = (last_frame - first_frame).abs()
        lower_motion = motion[:, :, lower_half, :].mean(dim=(1, 2, 3))
        
        # Detect support surface (consistent horizontal structure at bottom)
        bottom_region = video[:, :, :, -15:, :].mean(dim=(1, 2))  # Average over time and channels
        support_present = (bottom_region.std(dim=-1).mean(dim=-1) < 0.3).float()
        
        # Physics prior: low motion + support = stable
        motion_score = torch.exp(-lower_motion * 5)  # High score = low motion = stable
        stability = motion_score * support_present * 0.9 + 0.05
        
        return stability.unsqueeze(1), {
            'motion': lower_motion,
            'support': support_present,
            'prior_weight': self.prior_weight.item()
        }


class StabilityPredictor(nn.Module):
    """
    Predict stability from video.
    Can optionally use physics priors.
    """
    
    def __init__(self, use_prior: bool = False, hidden_dim: int = 128):
        super().__init__()
        self.use_prior = use_prior
        self.encoder = VideoEncoder(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        if use_prior:
            self.gravity_prior = GravityPrior()
    
    def forward(self, video: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            video: (B, T, C, H, W)
        Returns:
            prediction: (B, 1) stability probability
            info: dict with diagnostics
        """
        features = self.encoder(video)
        learned = torch.sigmoid(self.classifier(features))
        
        info = {'learned': learned.mean().item()}
        
        if self.use_prior:
            prior_pred, prior_info = self.gravity_prior(video)
            w = self.gravity_prior.prior_weight.clamp(max=0.8)
            prediction = w * prior_pred + (1 - w) * learned
            info.update(prior_info)
            info['blend_weight'] = w.item()
        else:
            prediction = learned
        
        return prediction, info


# %% [markdown]
# ## 3. Training and Evaluation
# 
# We train both models with varying amounts of data to test sample efficiency.

# %% [code]
def train_model(
    model: nn.Module,
    train_data: Tuple[torch.Tensor, torch.Tensor],
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 0.001,
    device: str = 'cpu'
) -> List[float]:
    """Train model and return loss history."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    videos, labels = train_data
    videos = videos.to(device)
    labels = labels.to(device)
    n_samples = len(labels)
    
    losses = []
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_samples)
        epoch_loss = 0.0
        n_batches = 0
        
        for i in range(0, n_samples, batch_size):
            idx = perm[i:min(i + batch_size, n_samples)]
            batch_videos = videos[idx]
            batch_labels = labels[idx].unsqueeze(1)
            
            pred, _ = model(batch_videos)
            loss = F.binary_cross_entropy(pred, batch_labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        losses.append(epoch_loss / n_batches)
    
    return losses


def evaluate_model(
    model: nn.Module,
    test_data: Tuple[torch.Tensor, torch.Tensor],
    device: str = 'cpu'
) -> float:
    """Evaluate model accuracy."""
    model = model.to(device)
    model.eval()
    
    videos, labels = test_data
    videos = videos.to(device)
    labels = labels.to(device)
    
    with torch.no_grad():
        pred, info = model(videos)
        pred_binary = (pred.squeeze() > 0.5).float()
        accuracy = (pred_binary == labels).float().mean().item()
    
    return accuracy, info


# %% [code]
# Main experiment: Compare with/without priors at different data sizes
print("="*70)
print("EXPERIMENT: Sample Efficiency with Physics Priors")
print("="*70)

train_sizes = [25, 50, 100, 200, 500]
n_seeds = 3
epochs = 50

results = {
    True: {n: [] for n in train_sizes},   # With prior
    False: {n: [] for n in train_sizes}   # Without prior
}

for seed in range(n_seeds):
    print(f"\nSeed {seed + 1}/{n_seeds}")
    
    for n_train in train_sizes:
        # Use subset of training data
        subset_videos = train_dataset.videos[:n_train]
        subset_labels = train_dataset.labels[:n_train]
        train_data = (subset_videos, subset_labels)
        test_data = (test_dataset.videos, test_dataset.labels)
        
        for use_prior in [True, False]:
            torch.manual_seed(seed * 1000 + n_train)
            
            model = StabilityPredictor(use_prior=use_prior)
            losses = train_model(model, train_data, epochs=epochs, device=device)
            accuracy, info = evaluate_model(model, test_data, device=device)
            
            results[use_prior][n_train].append(accuracy)
            
            prior_str = "w/ prior" if use_prior else "baseline"
            print(f"  N={n_train:3d}, {prior_str}: {accuracy:.1%}")

# %% [markdown]
# ## 4. Results Analysis

# %% [code]
print("\n" + "="*70)
print("RESULTS: Sample Efficiency Comparison")
print("="*70)
print(f"\n{'N_train':<10} {'With Prior':<18} {'Without Prior':<18} {'Δ':<10} {'Effect'}")
print("-"*70)

advantages = []
for n in train_sizes:
    with_prior = np.array(results[True][n])
    without_prior = np.array(results[False][n])
    
    diff = with_prior.mean() - without_prior.mean()
    advantages.append(diff)
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt((with_prior.std()**2 + without_prior.std()**2) / 2 + 0.001)
    d = diff / pooled_std
    
    if diff > 0.03:
        marker = "✅ PRIOR HELPS"
    elif diff > -0.02:
        marker = "⚠️ marginal"
    else:
        marker = "❌ no benefit"
    
    print(f"{n:<10} {with_prior.mean():.1%} ± {with_prior.std():.1%}   "
          f"{without_prior.mean():.1%} ± {without_prior.std():.1%}   "
          f"{diff:+.1%}     {marker}")

# Summary statistics
print("\n" + "-"*70)
small_data_adv = np.mean(advantages[:2])  # N=25, 50
large_data_adv = np.mean(advantages[2:])  # N=100, 200, 500

print(f"\nSmall data (N≤50) advantage: {small_data_adv:+.1%}")
print(f"Large data (N>50) advantage: {large_data_adv:+.1%}")

print("\n" + "="*70)
if small_data_adv > 0.02 and small_data_adv > large_data_adv:
    print("✅ HYPOTHESIS SUPPORTED: Physics priors improve sample efficiency")
    print("   Advantage is larger in low-data regime")
elif small_data_adv > 0:
    print("⚠️ MARGINAL SUPPORT: Small advantage in low-data regime")
else:
    print("❌ HYPOTHESIS NOT SUPPORTED: Priors did not help in this configuration")
print("="*70)

# %% [code]
# Plot results
plt.figure(figsize=(10, 6))

x = np.array(train_sizes)
with_prior_means = [np.mean(results[True][n]) for n in train_sizes]
with_prior_stds = [np.std(results[True][n]) for n in train_sizes]
without_prior_means = [np.mean(results[False][n]) for n in train_sizes]
without_prior_stds = [np.std(results[False][n]) for n in train_sizes]

plt.errorbar(x, with_prior_means, yerr=with_prior_stds, 
             label='With Physics Prior', marker='o', capsize=5, linewidth=2)
plt.errorbar(x, without_prior_means, yerr=without_prior_stds, 
             label='Without Prior (Baseline)', marker='s', capsize=5, linewidth=2)

plt.xlabel('Number of Training Samples', fontsize=12)
plt.ylabel('Test Accuracy', fontsize=12)
plt.title('NSCA Sample Efficiency: Physics Priors on Video Prediction', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.xscale('log')

# Shade the region where priors should help most
plt.axvspan(25, 100, alpha=0.1, color='green', label='Low-data regime')

plt.tight_layout()
plt.savefig('nsca_sample_efficiency_results.png', dpi=150)
plt.show()

print("\nPlot saved to 'nsca_sample_efficiency_results.png'")

# %% [markdown]
# ## 5. Conclusions
# 
# ### Key Findings:
# 
# 1. **Physics priors on dynamics**: When priors encode temporal physics 
#    (gravity, motion), they can provide sample efficiency benefits.
# 
# 2. **Prior accuracy matters**: The prior must be MORE accurate than 
#    what the network can learn independently in the low-data regime.
# 
# 3. **Task alignment**: Priors help most when the task requires 
#    understanding physics (prediction), not just pattern matching (classification).
# 
# ### Next Steps:
# 
# - Test on real Physion benchmark data
# - Evaluate with more complex physics scenarios
# - Measure prior weight adaptation during training

# %% [code]
print("\n" + "="*70)
print("EXPERIMENT COMPLETE")
print("="*70)
print(f"\nDevice used: {device}")
print("Upload this notebook to Kaggle for GPU-accelerated training.")
print("\nTo use real Physion data:")
print("  1. Download from: https://github.com/cogtoolslab/physics-benchmarking-neurips2021")
print("  2. Replace SyntheticPhysionDataset with real data loader")
print("  3. Re-run experiments")
