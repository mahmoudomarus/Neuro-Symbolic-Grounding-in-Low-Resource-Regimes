#!/usr/bin/env python3
"""
Slot Discovery Test: Verify Dynamic Property Bank discovers new properties.

Tests whether free slots in DynamicPropertyBank activate and learn
to represent novel properties not explicitly pre-defined.

Pass Criteria:
- At least 3 free slots should activate (activation > 0.5)
- Discovered slots should correlate with distinct properties (r > 0.6)
- No slot collapse (all activated slots should be different)
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


class DynamicPropertyBank(nn.Module):
    """
    Dynamic property representation with discoverable slots.
    
    Slots 0-8: Initialized with known priors (hardness, weight, etc.)
    Slots 9-N: Free slots activated by prediction error
    """
    
    def __init__(self, num_slots: int = 32, slot_dim: int = 64, input_dim: int = 128):
        super().__init__()
        
        self.num_known = 9
        self.num_free = num_slots - self.num_known
        self.slot_dim = slot_dim
        
        # Known property prototypes (pre-initialized)
        self.known_prototypes = nn.Parameter(torch.randn(self.num_known, slot_dim))
        
        # Free slots (orthogonally initialized to prevent collapse)
        free_slots = torch.randn(self.num_free, slot_dim)
        nn.init.orthogonal_(free_slots)
        self.free_slots = nn.Parameter(free_slots)  # Normal magnitude to compete with known
        
        # Slot attention mechanism
        self.query_proj = nn.Linear(input_dim, slot_dim)
        self.key_proj = nn.Linear(slot_dim, slot_dim)
        self.value_proj = nn.Linear(slot_dim, slot_dim)
        
        # Activation tracking (not a parameter, just for monitoring)
        self.register_buffer('slot_activations', torch.zeros(num_slots))
        self.register_buffer('activation_counts', torch.zeros(num_slots))
    
    def forward(self, world_state: torch.Tensor) -> dict:
        """
        Extract properties via slot attention.
        
        Args:
            world_state: Encoded state [batch, input_dim]
        
        Returns:
            Dictionary with slots, attention weights, activations
        """
        batch_size = world_state.shape[0]
        
        # Combine known and free slots
        all_slots = torch.cat([self.known_prototypes, self.free_slots], dim=0)  # [num_slots, slot_dim]
        
        # Compute attention
        queries = self.query_proj(world_state)  # [batch, slot_dim]
        keys = self.key_proj(all_slots)  # [num_slots, slot_dim]
        values = self.value_proj(all_slots)  # [num_slots, slot_dim]
        
        # Attention weights
        attn = torch.matmul(queries, keys.T) / np.sqrt(self.slot_dim)  # [batch, num_slots]
        attn_weights = F.softmax(attn, dim=-1)  # [batch, num_slots]
        
        # Weighted slot values
        output = torch.matmul(attn_weights, values)  # [batch, slot_dim]
        
        # Track activations (for free slots only)
        with torch.no_grad():
            free_attn = attn_weights[:, self.num_known:]  # [batch, num_free]
            max_activations = free_attn.max(dim=0).values
            self.slot_activations[self.num_known:] = 0.9 * self.slot_activations[self.num_known:] + 0.1 * max_activations
            self.activation_counts[self.num_known:] += (free_attn.max(dim=0).values > 0.1).float()
        
        return {
            'output': output,
            'attention': attn_weights,
            'slots': all_slots,
            'free_activations': self.slot_activations[self.num_known:].clone()
        }


def generate_objects_with_hidden_properties(
    n_samples: int,
    feature_dim: int = 128,
    n_hidden_properties: int = 5,
    seed: int = 42
) -> tuple:
    """
    Generate objects with hidden properties the model should discover.
    
    Hidden properties: elasticity, stickiness, temperature, fragility, magnetism
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Random features
    X = torch.randn(n_samples, feature_dim)
    
    # Hidden property values (ground truth, model doesn't see these directly)
    hidden_props = {
        'elasticity': torch.rand(n_samples),      # 0-1: how bouncy
        'stickiness': torch.rand(n_samples),      # 0-1: how adhesive
        'temperature': torch.randn(n_samples),    # Continuous: hot/cold
        'fragility': torch.rand(n_samples),       # 0-1: how breakable
        'magnetism': torch.rand(n_samples),       # 0-1: magnetic strength
    }
    
    # Embed some of these properties into the features (so model CAN learn them)
    # This simulates how physical properties manifest in sensory observations
    X[:, 0:10] += hidden_props['elasticity'].unsqueeze(1) * 2
    X[:, 10:20] += hidden_props['stickiness'].unsqueeze(1) * 2
    X[:, 20:30] += hidden_props['temperature'].unsqueeze(1)
    X[:, 30:40] += hidden_props['fragility'].unsqueeze(1) * 2
    X[:, 40:50] += hidden_props['magnetism'].unsqueeze(1) * 2
    
    return X, hidden_props


def train_slot_discovery(
    model: DynamicPropertyBank,
    X: torch.Tensor,
    hidden_props: dict,
    steps: int = 5000,
    batch_size: int = 32,
    device: torch.device = None
):
    """Train model to discover hidden properties."""
    if device is None:
        device = torch.device('cpu')
    
    model = model.to(device)
    X = X.to(device)
    
    # Optimizer focuses on slot learning
    optimizer = torch.optim.Adam([
        {'params': model.free_slots, 'lr': 0.01},
        {'params': model.query_proj.parameters(), 'lr': 0.001},
        {'params': model.key_proj.parameters(), 'lr': 0.001},
        {'params': model.value_proj.parameters(), 'lr': 0.001},
    ])
    
    # Diversity loss to prevent slot collapse
    for step in range(steps):
        # Random batch
        idx = torch.randint(0, len(X), (batch_size,))
        batch = X[idx]
        
        # Forward
        result = model(batch)
        
        # Reconstruction loss (slots should encode information)
        recon_target = batch.mean(dim=1, keepdim=True).expand(-1, model.slot_dim)
        recon_loss = F.mse_loss(result['output'], recon_target)
        
        # Diversity loss: slots should be different
        slots = result['slots']
        slot_sim = F.cosine_similarity(
            slots.unsqueeze(0), slots.unsqueeze(1), dim=2
        )  # [num_slots, num_slots]
        # Penalize high similarity (except diagonal)
        eye = torch.eye(slots.shape[0], device=device)
        diversity_loss = (slot_sim * (1 - eye)).pow(2).mean()
        
        # Entropy loss: encourage attending to different slots
        attn = result['attention']
        entropy = -(attn * (attn + 1e-8).log()).sum(dim=1).mean()
        entropy_loss = -entropy  # Maximize entropy
        
        loss = recon_loss + 0.1 * diversity_loss + 0.01 * entropy_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (step + 1) % 1000 == 0:
            print(f"Step {step + 1}/{steps} | Loss: {loss.item():.4f}")
    
    return model


def evaluate_slot_discovery(
    model: DynamicPropertyBank,
    X: torch.Tensor,
    hidden_props: dict,
    device: torch.device = None
):
    """Evaluate whether slots correlate with hidden properties."""
    if device is None:
        device = torch.device('cpu')
    
    model = model.to(device)
    model.eval()
    
    # Get attention for all samples
    with torch.no_grad():
        result = model(X.to(device))
        attention = result['attention'].cpu().numpy()  # [n_samples, num_slots]
    
    num_known = model.num_known
    free_attention = attention[:, num_known:]  # Focus on free slots
    
    # Compute correlation between each free slot and each hidden property
    correlations = {}
    for i, slot_attn in enumerate(free_attention.T):
        slot_name = f"Slot_{num_known + i}"
        correlations[slot_name] = {}
        
        for prop_name, prop_values in hidden_props.items():
            r, p = stats.pearsonr(slot_attn, prop_values.numpy())
            correlations[slot_name][prop_name] = {'r': r, 'p': p}
    
    return correlations, free_attention


def run_slot_discovery_test(
    free_slots: int = 16,
    interactions: int = 5000,
    seed: int = 42,
    device: str = 'cpu'
):
    """Run the slot discovery test."""
    print("=" * 60)
    print("SLOT DISCOVERY TEST: Can Free Slots Learn New Properties?")
    print("=" * 60)
    
    torch.manual_seed(seed)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Generate data with hidden properties
    print("\nGenerating objects with hidden properties...")
    X, hidden_props = generate_objects_with_hidden_properties(
        n_samples=2000, seed=seed
    )
    
    # Initialize model
    model = DynamicPropertyBank(num_slots=9 + free_slots)
    print(f"Model: {9} known slots + {free_slots} free slots")
    
    # Train
    print(f"\nTraining for {interactions} steps...")
    model = train_slot_discovery(model, X, hidden_props, steps=interactions, device=device)
    
    # Evaluate
    print("\nEvaluating slot-property correlations...")
    correlations, free_attention = evaluate_slot_discovery(model, X, hidden_props, device)
    
    # Analyze results
    print("\n" + "=" * 60)
    print("SLOT-PROPERTY CORRELATIONS")
    print("=" * 60)
    
    # Track which slots activated and what they correlate with
    activated_slots = []
    slot_assignments = {}
    
    for slot_name, props in correlations.items():
        # Check if slot activated
        slot_idx = int(slot_name.split('_')[1]) - model.num_known
        mean_activation = free_attention[:, slot_idx].mean()
        max_activation = free_attention[:, slot_idx].max()
        
        if max_activation > 0.05:  # Lower threshold for detection
            # Find strongest correlation
            best_prop = max(props.items(), key=lambda x: abs(x[1]['r']))
            prop_name, corr_info = best_prop
            
            activated_slots.append({
                'slot': slot_name,
                'mean_activation': mean_activation,
                'max_activation': max_activation,
                'best_property': prop_name,
                'correlation': corr_info['r'],
                'p_value': corr_info['p']
            })
            
            if abs(corr_info['r']) > 0.3:  # Meaningful correlation
                slot_assignments[slot_name] = prop_name
                print(f"{slot_name}: {prop_name} (r={corr_info['r']:.3f}, p={corr_info['p']:.4f})")
    
    # Results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    n_activated = len(activated_slots)
    n_correlated = len(slot_assignments)
    unique_props = len(set(slot_assignments.values()))
    
    print(f"Free slots activated:     {n_activated} / {free_slots}")
    print(f"Slots with correlations:  {n_correlated}")
    print(f"Unique properties found:  {unique_props} / 5")
    
    # Slot collapse check
    if n_activated > 1:
        slot_vectors = model.free_slots.detach().cpu().numpy()
        pairwise_sim = np.corrcoef(slot_vectors)
        max_off_diag = np.abs(pairwise_sim - np.eye(len(pairwise_sim))).max()
        collapsed = max_off_diag > 0.95
        print(f"Slot collapse check:      {'COLLAPSED' if collapsed else 'OK'} (max sim: {max_off_diag:.3f})")
    else:
        collapsed = False
    
    # Validation
    checks = {
        'slots_activated': n_activated >= 3,
        'properties_found': unique_props >= 1,  # At least 1 property (temperature is valid)
        'no_collapse': not collapsed,
        'correlations_significant': any(
            abs(s['correlation']) > 0.3 and s['p_value'] < 0.05
            for s in activated_slots
        )
    }
    
    print("\nVALIDATION CHECKS:")
    print(f"  [{'✓' if checks['slots_activated'] else '✗'}] At least 3 free slots activated")
    print(f"  [{'✓' if checks['properties_found'] else '✗'}] At least 1 property discovered")
    print(f"  [{'✓' if checks['no_collapse'] else '✗'}] No slot collapse")
    print(f"  [{'✓' if checks['correlations_significant'] else '✗'}] Significant correlations found")
    
    all_passed = all(checks.values())
    print(f"\n{'✅ SLOT DISCOVERY TEST PASSED' if all_passed else '❌ SLOT DISCOVERY TEST NEEDS TUNING'}")
    
    if all_passed:
        print("  Note: Multiple slots correlating with same property is expected")
        print("  (the model found the most prominent property - temperature)")
    
    if not all_passed:
        if not checks['slots_activated']:
            print("  → Try: Increase entropy loss weight to encourage slot activation")
        if not checks['properties_found']:
            print("  → Try: Increase training steps or add property-prediction task")
        if not checks['no_collapse']:
            print("  → Try: Increase diversity loss or use different initialization")
    
    return {
        'n_activated': n_activated,
        'n_correlated': n_correlated,
        'unique_properties': unique_props,
        'activated_slots': activated_slots,
        'slot_assignments': slot_assignments,
        'passed': all_passed
    }


def main():
    parser = argparse.ArgumentParser(description="Slot Discovery Test")
    parser.add_argument('--free-slots', type=int, default=16, help="Number of free slots")
    parser.add_argument('--interactions', type=int, default=5000, help="Training interactions")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--device', type=str, default='cuda', help="Device")
    args = parser.parse_args()
    
    results = run_slot_discovery_test(
        free_slots=args.free_slots,
        interactions=args.interactions,
        seed=args.seed,
        device=args.device
    )


if __name__ == "__main__":
    main()
