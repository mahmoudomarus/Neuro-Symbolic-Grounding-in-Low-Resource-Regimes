"""
NSCA Physion Prediction Test
============================

This tests the CORE hypothesis correctly:
- Prior helps the WORLD MODEL predict outcomes
- NOT the policy (muscle movements)

Task: Given video frames of objects, predict "Will it fall?"
- With prior: Model has bias toward "falls if unsupported"
- Without prior: Model must learn physics from scratch

Run on Kaggle with free P100 GPU.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
import os

# ============================================================
# PART 1: PHYSICS PRIORS FOR WORLD MODEL (not policy!)
# ============================================================

class SupportPrior(nn.Module):
    """
    Innate prior: Unsupported objects fall.
    
    This is applied to the WORLD MODEL's prediction,
    NOT to the action output.
    
    Input: Object positions/features
    Output: Bias toward "will fall" if unsupported
    """
    
    def __init__(self, feature_dim: int = 256):
        super().__init__()
        
        # Learn to detect "support" from visual features
        self.support_detector = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # 0 = unsupported, 1 = supported
        )
        
        # Prior strength (learnable, can be overridden)
        self._prior_weight = nn.Parameter(torch.tensor(0.6))  # Start believing prior
        self.min_prior_weight = 0.2
        
    @property
    def prior_weight(self) -> torch.Tensor:
        """Effective prior weight with floor."""
        return self.min_prior_weight + F.softplus(self._prior_weight)
    
    def forward(self, object_features: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Predict "support" level for object.
        
        Returns:
            support_score: 0 = will fall, 1 = stable
            diagnostics: dict with prior contribution
        """
        # Learned support detection
        learned_support = self.support_detector(object_features)
        
        # Prior belief: things without visible support fall
        # (This is the "innate" part - we bias toward falling)
        prior_bias = 0.3  # Prior says "probably falls" (low support)
        
        # Blend learned and prior
        w = self.prior_weight
        support_score = w * prior_bias + (1 - w) * learned_support
        
        return support_score, {
            'prior_weight': w.item(),
            'learned_support': learned_support.mean().item(),
            'prior_bias': prior_bias
        }


class GravityPriorWorldModel(nn.Module):
    """
    Prior: Objects accelerate downward at ~9.8 m/s².
    
    Applied to TRAJECTORY PREDICTION, not action selection.
    """
    
    def __init__(self, feature_dim: int = 256):
        super().__init__()
        
        self.gravity_vector = torch.tensor([0.0, -9.8, 0.0])  # y-down
        
        # Learnable correction for exceptions (balloons, magnets)
        self.correction_net = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # dx, dy, dz correction
        )
        
        self._prior_weight = nn.Parameter(torch.tensor(0.6))
        self.min_prior_weight = 0.2
    
    @property
    def prior_weight(self) -> torch.Tensor:
        return self.min_prior_weight + F.softplus(self._prior_weight)
    
    def predict_trajectory(
        self, 
        object_features: torch.Tensor,
        current_velocity: torch.Tensor,
        dt: float = 0.1
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Predict where object will be after dt seconds.
        
        Prior: velocity += gravity * dt
        Learned: correction for special cases
        """
        batch_size = object_features.shape[0]
        gravity = self.gravity_vector.to(object_features.device)
        gravity = gravity.unsqueeze(0).expand(batch_size, -1)
        
        # Prior prediction: just gravity
        prior_delta_v = gravity * dt
        
        # Learned correction
        learned_correction = self.correction_net(object_features)
        
        # Blend
        w = self.prior_weight
        delta_v = w * prior_delta_v + (1 - w) * learned_correction
        
        # New velocity
        new_velocity = current_velocity + delta_v
        
        return new_velocity, {
            'prior_weight': w.item(),
            'gravity_contribution': (w * prior_delta_v).mean().item()
        }


# ============================================================
# PART 2: VISUAL ENCODER WITH GABOR PRIORS
# ============================================================

class VisualEncoderWithPriors(nn.Module):
    """
    Vision encoder with Gabor filter priors.
    
    The Gabor filters help extract edges/orientations faster,
    which helps detect "support" (edges touching = supported).
    """
    
    def __init__(self, output_dim: int = 256):
        super().__init__()
        
        # Gabor-initialized first layer (the "prior")
        self.gabor_conv = nn.Conv2d(3, 32, kernel_size=7, padding=3)
        self._init_gabor_filters()
        
        # Learnable layers
        self.conv_layers = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.fc = nn.Linear(256, output_dim)
    
    def _init_gabor_filters(self):
        """Initialize first layer with Gabor filters (biological prior)."""
        with torch.no_grad():
            filters = []
            for theta in np.linspace(0, np.pi, 8):  # 8 orientations
                for sigma in [1, 2, 4, 8]:  # 4 scales
                    gabor = self._make_gabor(theta, sigma)
                    filters.append(gabor)
            
            # Stack and assign (32 filters = 8 orientations × 4 scales)
            filters = torch.stack(filters[:32])
            self.gabor_conv.weight.data = filters.unsqueeze(1).repeat(1, 3, 1, 1)
    
    def _make_gabor(self, theta: float, sigma: float) -> torch.Tensor:
        """Create single Gabor filter."""
        size = 7
        x = torch.arange(size) - size // 2
        y = torch.arange(size) - size // 2
        x, y = torch.meshgrid(x, y, indexing='ij')
        
        x_theta = x * np.cos(theta) + y * np.sin(theta)
        y_theta = -x * np.sin(theta) + y * np.cos(theta)
        
        gaussian = torch.exp(-(x_theta**2 + y_theta**2) / (2 * sigma**2))
        sinusoid = torch.cos(2 * np.pi * x_theta / sigma)
        
        gabor = gaussian * sinusoid
        return gabor.float()
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Extract features from image."""
        x = self.gabor_conv(image)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ============================================================
# PART 3: STABILITY PREDICTION MODEL (THE FULL TEST)
# ============================================================

class NSCAStabilityPredictor(nn.Module):
    """
    Full NSCA model for stability prediction.
    
    Task: Given image of objects, predict "Will it fall?"
    
    Architecture:
    1. Visual encoder (with Gabor priors) extracts features
    2. Support prior biases toward "falls if unsupported"
    3. Final prediction blends learned + prior
    """
    
    def __init__(self, use_priors: bool = True):
        super().__init__()
        
        self.use_priors = use_priors
        
        # Visual encoder
        self.encoder = VisualEncoderWithPriors(output_dim=256)
        
        # Physics priors (only if enabled)
        if use_priors:
            self.support_prior = SupportPrior(feature_dim=256)
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # 0 = falls, 1 = stable
        )
    
    def forward(self, image: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Predict stability from image.
        
        Returns:
            stability: 0 = will fall, 1 = will stay
            diagnostics: dict with prior contributions
        """
        # Extract visual features
        features = self.encoder(image)
        
        # Get learned prediction
        learned_stability = self.classifier(features)
        
        diagnostics = {'learned_stability': learned_stability.mean().item()}
        
        if self.use_priors:
            # Get prior-biased prediction
            prior_stability, prior_diag = self.support_prior(features)
            diagnostics.update(prior_diag)
            
            # Blend (prior influences final prediction)
            # Higher prior_weight = more "pessimistic" about stability
            final_stability = 0.5 * learned_stability + 0.5 * prior_stability
        else:
            final_stability = learned_stability
        
        return final_stability, diagnostics


# ============================================================
# PART 4: SYNTHETIC DATASET (for testing without Physion)
# ============================================================

class SyntheticStabilityDataset:
    """
    Generate simple stability prediction examples.
    
    For real testing, use Physion dataset.
    This is just for local validation.
    """
    
    def __init__(self, n_samples: int = 100):
        self.n_samples = n_samples
    
    def generate_batch(self, batch_size: int = 16) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate batch of (image, label) pairs.
        
        Image: Simple 64x64 with shapes
        Label: 1 if stable (supported), 0 if unstable (will fall)
        """
        images = []
        labels = []
        
        for _ in range(batch_size):
            img = torch.zeros(3, 64, 64)
            
            # Randomly decide if stable or not
            is_stable = np.random.random() > 0.5
            
            # Draw a "block"
            block_y = np.random.randint(20, 50)
            block_x = np.random.randint(20, 44)
            img[:, block_y:block_y+10, block_x:block_x+10] = 1.0  # White block
            
            if is_stable:
                # Add "ground" support below block
                img[1, 55:60, :] = 0.5  # Green ground
            else:
                # No support - block is floating
                pass
            
            images.append(img)
            labels.append(1.0 if is_stable else 0.0)
        
        return torch.stack(images), torch.tensor(labels).unsqueeze(1)


# ============================================================
# PART 5: TRAINING AND EVALUATION
# ============================================================

def train_and_evaluate(use_priors: bool, n_epochs: int = 50, device: str = 'cpu'):
    """
    Train stability predictor and evaluate.
    
    Compare: With priors vs Without priors
    """
    print(f"\n{'='*50}")
    print(f"Training with priors={use_priors}")
    print(f"{'='*50}")
    
    # Create model
    model = NSCAStabilityPredictor(use_priors=use_priors).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    dataset = SyntheticStabilityDataset()
    
    # Training
    losses = []
    for epoch in range(n_epochs):
        images, labels = dataset.generate_batch(batch_size=32)
        images, labels = images.to(device), labels.to(device)
        
        predictions, diag = model(images)
        loss = F.binary_cross_entropy(predictions, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: loss={loss.item():.4f}")
            if use_priors:
                print(f"  Prior weight: {diag.get('prior_weight', 'N/A')}")
    
    # Evaluation
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for _ in range(10):  # 10 batches
            images, labels = dataset.generate_batch(batch_size=32)
            images, labels = images.to(device), labels.to(device)
            
            predictions, _ = model(images)
            predicted_labels = (predictions > 0.5).float()
            
            correct += (predicted_labels == labels).sum().item()
            total += labels.size(0)
    
    accuracy = correct / total
    print(f"\nFinal accuracy: {accuracy:.1%}")
    
    return accuracy, losses


def run_comparison():
    """
    Run the key comparison: Priors vs No Priors
    """
    print("="*60)
    print("NSCA STABILITY PREDICTION TEST")
    print("="*60)
    print()
    print("HYPOTHESIS:")
    print("  Model WITH physics priors learns faster and better")
    print("  than model WITHOUT priors (learning from scratch)")
    print()
    print("TASK: Predict if block will fall (unsupported) or stay (supported)")
    print()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print()
    
    # Test with fewer epochs to see learning speed difference
    epochs_to_test = [10, 25, 50]
    
    results = {'with_priors': [], 'without_priors': []}
    
    for n_epochs in epochs_to_test:
        print(f"\n--- Testing with {n_epochs} epochs ---")
        
        acc_with, _ = train_and_evaluate(use_priors=True, n_epochs=n_epochs, device=device)
        acc_without, _ = train_and_evaluate(use_priors=False, n_epochs=n_epochs, device=device)
        
        results['with_priors'].append(acc_with)
        results['without_priors'].append(acc_without)
    
    # Summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"\n{'Epochs':<10} {'With Priors':<15} {'Without Priors':<15} {'Diff'}")
    print("-"*50)
    
    for i, epochs in enumerate(epochs_to_test):
        acc_w = results['with_priors'][i]
        acc_wo = results['without_priors'][i]
        diff = acc_w - acc_wo
        winner = "✅" if diff > 0 else "❌"
        print(f"{epochs:<10} {acc_w:<15.1%} {acc_wo:<15.1%} {diff:+.1%} {winner}")
    
    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)
    
    # Check if priors helped at low epochs (sample efficiency)
    early_advantage = results['with_priors'][0] - results['without_priors'][0]
    
    if early_advantage > 0.05:
        print("✅ PRIORS HELP: Faster learning in early epochs")
        print(f"   At {epochs_to_test[0]} epochs: +{early_advantage:.1%} advantage")
    else:
        print("⚠️  INCONCLUSIVE: Priors didn't show clear advantage")
    
    return results


if __name__ == "__main__":
    run_comparison()
