#!/usr/bin/env python3
"""
Catastrophic Forgetting Test for EWC Validation.

Tests whether Elastic Weight Consolidation prevents forgetting
of previously learned tasks.

Protocol:
1. Train on Task A → measure performance
2. Train on Task B → measure performance
3. Re-test Task A:
   - Without EWC: expect ~30% (catastrophic forgetting)
   - With EWC: expect ~70% (mild forgetting, acceptable)
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


class SimpleClassifier(nn.Module):
    """Simple classifier for forgetting test."""
    
    def __init__(self, input_dim: int = 64, hidden_dim: int = 256, output_dim: int = 5):
        super().__init__()
        # Simpler architecture without BatchNorm (avoids train/eval mode issues)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EWCRegularizer:
    """Elastic Weight Consolidation implementation."""
    
    def __init__(self, model: nn.Module, ewc_weight: float = 1000.0):
        self.model = model
        self.ewc_weight = ewc_weight
        self.fisher = {}
        self.old_params = {}
    
    def compute_fisher(self, dataloader: DataLoader, device: torch.device):
        """Compute diagonal Fisher Information Matrix."""
        self.model.eval()
        
        # Initialize Fisher
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.fisher[name] = torch.zeros_like(param)
                self.old_params[name] = param.data.clone()
        
        # Accumulate gradients
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            
            self.model.zero_grad()
            output = self.model(x)
            loss = F.cross_entropy(output, y)
            loss.backward()
            
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    self.fisher[name] += param.grad.data ** 2
        
        # Normalize
        n_samples = len(dataloader.dataset)
        for name in self.fisher:
            self.fisher[name] /= n_samples
    
    def penalty(self) -> torch.Tensor:
        """Compute EWC penalty."""
        loss = 0.0
        for name, param in self.model.named_parameters():
            if name in self.fisher:
                loss += (self.fisher[name] * (param - self.old_params[name]) ** 2).sum()
        return self.ewc_weight * loss


def generate_task_data(
    task_id: int,
    n_samples: int = 1000,
    n_classes: int = 5,  # Reduced from 10 for easier learning
    input_dim: int = 64,
    seed: int = 42,
    is_test: bool = False
) -> tuple:
    """Generate synthetic data for a task."""
    # Use different seed for train vs test, but same cluster structure
    actual_seed = seed + task_id * 100 + (1000 if is_test else 0)
    np.random.seed(seed + task_id * 100)  # Same centers for train/test
    torch.manual_seed(actual_seed)
    
    # Create well-separated cluster centers
    centers = np.zeros((n_classes, input_dim))
    for i in range(n_classes):
        # Each class gets a unique direction
        direction = np.zeros(input_dim)
        if task_id == 0:
            # Task A: classes differ in first 32 dims
            start_idx = (i * 6) % 32
            direction[start_idx:start_idx+6] = 5.0
        else:
            # Task B: classes differ in last 32 dims  
            start_idx = 32 + (i * 6) % 32
            direction[start_idx:start_idx+6] = 5.0
        centers[i] = direction
    
    # Generate samples
    np.random.seed(actual_seed)  # Different samples for train/test
    X = []
    y = []
    
    for _ in range(n_samples):
        label = np.random.randint(0, n_classes)
        # Add noise but keep clusters separable
        sample = centers[label] + np.random.randn(input_dim) * 0.5
        X.append(sample)
        y.append(label)
    
    X = torch.tensor(np.array(X), dtype=torch.float32)
    y = torch.tensor(np.array(y), dtype=torch.long)
    
    return X, y


def train_on_task(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int = 20,
    ewc: EWCRegularizer = None,
    task_name: str = "Task"
) -> float:
    """Train model on a task and return test accuracy."""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Higher LR
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=epochs//2, gamma=0.5)
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        correct_train = 0
        total_train = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            output = model(x)
            loss = F.cross_entropy(output, y)
            
            # Add EWC penalty if available
            if ewc is not None and ewc.fisher:
                loss += ewc.penalty()
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
            # Track training accuracy
            pred = output.argmax(dim=1)
            correct_train += (pred == y).sum().item()
            total_train += len(y)
        
        scheduler.step()
    
    # Evaluate
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            pred = output.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += len(y)
    
    accuracy = 100.0 * correct / total
    train_acc = 100.0 * correct_train / total_train
    print(f"  {task_name} train: {train_acc:.1f}%, test: {accuracy:.1f}%")
    return accuracy


def run_forgetting_test(
    ewc_weight: float = 1000.0,
    use_ewc: bool = True,
    n_train: int = 500,
    n_test: int = 200,
    epochs: int = 20,
    seed: int = 42,
    device: str = 'cpu'
):
    """Run the catastrophic forgetting test."""
    print("=" * 60)
    print(f"FORGETTING TEST (EWC {'ON' if use_ewc else 'OFF'})")
    print("=" * 60)
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Generate data for two tasks
    X_train_A, y_train_A = generate_task_data(0, n_train, seed=seed, is_test=False)
    X_test_A, y_test_A = generate_task_data(0, n_test, seed=seed, is_test=True)
    
    X_train_B, y_train_B = generate_task_data(1, n_train, seed=seed, is_test=False)
    X_test_B, y_test_B = generate_task_data(1, n_test, seed=seed, is_test=True)
    
    # DataLoaders
    train_loader_A = DataLoader(TensorDataset(X_train_A, y_train_A), batch_size=32, shuffle=True)
    test_loader_A = DataLoader(TensorDataset(X_test_A, y_test_A), batch_size=32)
    
    train_loader_B = DataLoader(TensorDataset(X_train_B, y_train_B), batch_size=32, shuffle=True)
    test_loader_B = DataLoader(TensorDataset(X_test_B, y_test_B), batch_size=32)
    
    # Initialize model
    model = SimpleClassifier().to(device)
    ewc = EWCRegularizer(model, ewc_weight) if use_ewc else None
    
    # Phase 1: Train on Task A
    print("\n[Phase 1] Training on Task A...")
    acc_A_initial = train_on_task(model, train_loader_A, test_loader_A, device, epochs, task_name="Task A")
    
    # Compute Fisher for EWC
    if ewc:
        print("  Computing Fisher Information...")
        ewc.compute_fisher(train_loader_A, device)
    
    # Phase 2: Train on Task B
    print("\n[Phase 2] Training on Task B (while preserving Task A)...")
    acc_B = train_on_task(model, train_loader_B, test_loader_B, device, epochs, ewc, task_name="Task B")
    
    # Phase 3: Re-test Task A
    print("\n[Phase 3] Re-testing Task A (after learning Task B)...")
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in test_loader_A:
            x, y = x.to(device), y.to(device)
            output = model(x)
            pred = output.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += len(y)
    
    acc_A_final = 100.0 * correct / total
    
    # Results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Task A (initial):  {acc_A_initial:.1f}%")
    print(f"Task B:            {acc_B:.1f}%")
    print(f"Task A (after B):  {acc_A_final:.1f}%")
    print(f"Forgetting:        {acc_A_initial - acc_A_final:.1f}%")
    print("=" * 60)
    
    # Determine pass/fail
    if use_ewc:
        passed = acc_A_final >= 60.0  # Should retain >60%
        print(f"\nEWC TEST {'PASSED' if passed else 'FAILED'}: ", end="")
        print(f"Task A retention {acc_A_final:.1f}% (target: >60%)")
    else:
        expected_forgetting = acc_A_initial - acc_A_final > 30
        print(f"\nBASELINE (no EWC): ", end="")
        if expected_forgetting:
            print(f"Catastrophic forgetting confirmed ({acc_A_initial - acc_A_final:.1f}% drop)")
        else:
            print(f"Unexpectedly low forgetting ({acc_A_initial - acc_A_final:.1f}% drop)")
    
    return {
        'acc_A_initial': acc_A_initial,
        'acc_B': acc_B,
        'acc_A_final': acc_A_final,
        'forgetting': acc_A_initial - acc_A_final,
        'ewc_enabled': use_ewc,
        'passed': acc_A_final >= 60.0 if use_ewc else True
    }


def main():
    parser = argparse.ArgumentParser(description="Catastrophic Forgetting Test")
    parser.add_argument('--ewc-weight', type=float, default=1000.0, help="EWC regularization weight")
    parser.add_argument('--no-ewc', action='store_true', help="Disable EWC for baseline")
    parser.add_argument('--epochs', type=int, default=20, help="Training epochs per task")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--device', type=str, default='cuda', help="Device")
    args = parser.parse_args()
    
    # Run with EWC
    print("\n" + "=" * 70)
    print("RUNNING WITH EWC")
    print("=" * 70)
    results_ewc = run_forgetting_test(
        ewc_weight=args.ewc_weight,
        use_ewc=True,
        epochs=args.epochs,
        seed=args.seed,
        device=args.device
    )
    
    # Run without EWC (baseline)
    if not args.no_ewc:
        print("\n" + "=" * 70)
        print("RUNNING WITHOUT EWC (BASELINE)")
        print("=" * 70)
        results_no_ewc = run_forgetting_test(
            use_ewc=False,
            epochs=args.epochs,
            seed=args.seed,
            device=args.device
        )
        
        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Without EWC: {results_no_ewc['forgetting']:.1f}% forgetting")
        print(f"With EWC:    {results_ewc['forgetting']:.1f}% forgetting")
        print(f"Improvement: {results_no_ewc['forgetting'] - results_ewc['forgetting']:.1f}% less forgetting")
        
        if results_ewc['passed'] and results_no_ewc['forgetting'] > results_ewc['forgetting']:
            print("\n✅ EWC VALIDATION PASSED")
        else:
            print("\n❌ EWC VALIDATION NEEDS INVESTIGATION")


if __name__ == "__main__":
    main()
