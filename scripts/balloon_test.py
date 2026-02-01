#!/usr/bin/env python3
"""
Balloon Test: Verify that physics priors can be overridden.

Tests whether the AdaptivePhysicsPrior correctly learns to reduce
its prior weight when encountering anti-gravity scenarios (balloons).

Pass Criteria:
- prior_weight should drop from ~0.9 → ~0.35
- Model should predict upward motion after training
- prior_weight should not go below 0.3 (critical period protection)
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


class AdaptivePhysicsPrior(nn.Module):
    """
    Physics prior with learnable correction for exceptions.
    
    The prior starts strong (prior_weight ~0.9) but can be
    overridden by data, with a minimum floor at 0.3.
    """
    
    def __init__(self, feature_dim: int = 64, initial_prior_weight: float = 0.9):
        super().__init__()
        
        self.prior_gravity = -9.8  # Innate downward bias
        
        self.correction_net = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Predict vertical motion
        )
        
        # Initialize prior_weight using inverse softplus
        # We want: 0.3 + softplus(x - 0.3) = initial_prior_weight
        # Using a simpler formulation that allows more aggressive learning
        self._prior_weight = nn.Parameter(torch.tensor(initial_prior_weight))
        
        # Learnable scaling factor to allow faster adaptation
        self.adaptation_rate = nn.Parameter(torch.tensor(1.0))
    
    @property
    def effective_prior_weight(self) -> torch.Tensor:
        """Get effective prior weight with floor at 0.3."""
        # Sigmoid maps to (0,1), then scale to (0.3, 1.0)
        return 0.3 + 0.7 * torch.sigmoid(self._prior_weight)
    
    def forward(self, state: torch.Tensor, anti_gravity: bool = False) -> torch.Tensor:
        """
        Predict vertical motion.
        
        Args:
            state: Object state [batch, feature_dim]
            anti_gravity: If True, simulate balloon (for ground truth)
        
        Returns:
            Predicted vertical velocity
        """
        batch_size = state.shape[0]
        
        # Prior: gravity pulls down
        prior_motion = torch.full((batch_size, 1), self.prior_gravity, device=state.device)
        
        # Learned correction
        correction = self.correction_net(state)
        
        # Blend with learnable weight
        w = self.effective_prior_weight
        prediction = w * prior_motion + (1 - w) * correction
        
        return prediction


def generate_balloon_data(n_samples: int, feature_dim: int = 64, upward_velocity: float = 5.0):
    """
    Generate synthetic balloon data.
    
    Balloons rise (positive vertical velocity) instead of falling.
    """
    # Random state features
    X = torch.randn(n_samples, feature_dim)
    
    # Balloons go UP (anti-gravity)
    y = torch.full((n_samples, 1), upward_velocity)
    
    return X, y


def generate_normal_data(n_samples: int, feature_dim: int = 64, gravity: float = -9.8):
    """Generate normal falling object data."""
    X = torch.randn(n_samples, feature_dim)
    y = torch.full((n_samples, 1), gravity)
    return X, y


def run_balloon_test(
    steps: int = 1000,
    batch_size: int = 32,
    lr: float = 0.01,
    seed: int = 42,
    device: str = 'cpu',
    verbose: bool = True
):
    """Run the balloon (prior override) test."""
    print("=" * 60)
    print("BALLOON TEST: Can Physics Priors Be Overridden?")
    print("=" * 60)
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model = AdaptivePhysicsPrior().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    initial_weight = model.effective_prior_weight.item()
    print(f"\nInitial prior_weight: {initial_weight:.3f}")
    print(f"Training on BALLOON data (upward motion)...\n")
    
    # Track prior weight evolution
    weight_history = [initial_weight]
    loss_history = []
    
    # Training on balloon data
    for step in range(steps):
        # Generate batch of balloon data
        X, y = generate_balloon_data(batch_size)
        X, y = X.to(device), y.to(device)
        
        # Forward
        pred = model(X)
        loss = F.mse_loss(pred, y)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track
        current_weight = model.effective_prior_weight.item()
        weight_history.append(current_weight)
        loss_history.append(loss.item())
        
        if verbose and (step + 1) % 100 == 0:
            print(f"Step {step + 1:4d} | Loss: {loss.item():7.3f} | prior_weight: {current_weight:.3f}")
    
    # Final evaluation
    final_weight = model.effective_prior_weight.item()
    
    # Test prediction
    test_X = torch.randn(10, 64).to(device)
    with torch.no_grad():
        predictions = model(test_X)
    
    avg_prediction = predictions.mean().item()
    
    # Results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Initial prior_weight: {initial_weight:.3f}")
    print(f"Final prior_weight:   {final_weight:.3f}")
    print(f"Weight reduction:     {initial_weight - final_weight:.3f}")
    print(f"Final loss:           {loss_history[-1]:.3f}")
    print(f"Average prediction:   {avg_prediction:.2f} (target: +5.0)")
    print("=" * 60)
    
    # Validation checks
    checks = {
        'weight_reduced': final_weight < initial_weight - 0.1,
        'above_floor': final_weight >= 0.29,  # Allow tiny numerical error
        'predicts_upward': avg_prediction > 0,
        'learns_pattern': loss_history[-1] < loss_history[0] / 2
    }
    
    print("\nVALIDATION CHECKS:")
    print(f"  [{'✓' if checks['weight_reduced'] else '✗'}] Prior weight reduced significantly")
    print(f"  [{'✓' if checks['above_floor'] else '✗'}] Prior weight above floor (0.3)")
    print(f"  [{'✓' if checks['predicts_upward'] else '✗'}] Model predicts upward motion")
    print(f"  [{'✓' if checks['learns_pattern'] else '✗'}] Loss decreased over training")
    
    all_passed = all(checks.values())
    print(f"\n{'✅ BALLOON TEST PASSED' if all_passed else '❌ BALLOON TEST FAILED'}")
    
    if not checks['weight_reduced']:
        print("  → Prior weight stuck! Check gradient flow through softplus.")
    if not checks['above_floor']:
        print("  → Prior weight below floor! Critical period protection broken.")
    if not checks['predicts_upward']:
        print("  → Not predicting upward! correction_net not learning.")
    
    return {
        'initial_weight': initial_weight,
        'final_weight': final_weight,
        'weight_history': weight_history,
        'loss_history': loss_history,
        'avg_prediction': avg_prediction,
        'passed': all_passed
    }


def main():
    parser = argparse.ArgumentParser(description="Balloon (Prior Override) Test")
    parser.add_argument('--steps', type=int, default=1000, help="Training steps")
    parser.add_argument('--batch-size', type=int, default=32, help="Batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="Learning rate")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--device', type=str, default='cuda', help="Device")
    parser.add_argument('--anti-gravity', action='store_true', help="Enable anti-gravity (default)")
    parser.add_argument('--plot', action='store_true', help="Plot weight evolution")
    args = parser.parse_args()
    
    results = run_balloon_test(
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        device=args.device
    )
    
    if args.plot:
        try:
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Prior weight evolution
            ax1.plot(results['weight_history'])
            ax1.axhline(y=0.3, color='r', linestyle='--', label='Floor (0.3)')
            ax1.axhline(y=results['initial_weight'], color='g', linestyle='--', label='Initial')
            ax1.set_xlabel('Training Step')
            ax1.set_ylabel('Prior Weight')
            ax1.set_title('Prior Weight Adaptation')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Loss curve
            ax2.plot(results['loss_history'])
            ax2.set_xlabel('Training Step')
            ax2.set_ylabel('Loss')
            ax2.set_title('Training Loss')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('balloon_test_results.png', dpi=150)
            print(f"\nPlot saved to balloon_test_results.png")
            plt.show()
        except ImportError:
            print("matplotlib not available for plotting")


if __name__ == "__main__":
    main()
