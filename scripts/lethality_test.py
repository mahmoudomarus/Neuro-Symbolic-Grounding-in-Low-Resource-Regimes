#!/usr/bin/env python3
"""
Lethality Test: Minimal Viable Experiment for NSCA

This script tests the CORE HYPOTHESIS before any expensive training:
"Do adaptive physics priors improve sample efficiency?"

Cost: ~$5 on Vast.ai (2-3 hours)
Success: prior_weight=0.9 beats prior_weight=0.5 by >20%

Run this BEFORE downloading ImageNet or any expensive compute.
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import metaworld, fall back to simulation
try:
    import metaworld
    import gymnasium as gym
    HAS_METAWORLD = True
except ImportError:
    HAS_METAWORLD = False
    print("WARNING: MetaWorld not installed. Using simulated environment.")


@dataclass
class LethalityConfig:
    """Configuration for the lethality test."""
    task: str = "pick-place-v3"  # MetaWorld v3 naming
    n_seeds: int = 5
    n_demos: int = 5
    training_steps: int = 5000
    eval_episodes: int = 20
    conditions: Tuple[float, ...] = (0.9, 0.5, 0.1)  # prior_weight values
    output_dir: str = "results/lethality_test"


class AdaptivePhysicsPriorSimple(nn.Module):
    """
    Simplified Adaptive Physics Prior for state-vector inputs.
    
    This tests the CORE mechanism without vision/audio complexity.
    """
    
    def __init__(self, state_dim: int, action_dim: int, prior_weight_init: float = 0.9):
        super().__init__()
        
        self.gravity_prior = torch.tensor([0.0, 0.0, -9.8])  # z-down gravity
        
        # Learnable prior weight with softplus constraint
        # To get effective_weight = prior_weight_init, we need:
        # min_weight + softplus(x) = prior_weight_init
        # x = log(exp(prior_weight_init - min_weight) - 1)  (inverse softplus)
        self.min_prior_weight = 0.3
        target = prior_weight_init - self.min_prior_weight
        # inverse_softplus: x = log(exp(y) - 1)
        init_raw = np.log(np.exp(target) - 1) if target > 0 else -10.0
        self._prior_weight = nn.Parameter(torch.tensor(init_raw, dtype=torch.float32))
        
        # Correction network learns residual
        self.correction_net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # Predict dx, dy, dz correction
        )
        
        # Policy head
        self.policy = nn.Sequential(
            nn.Linear(state_dim + 3, 128),  # state + physics prediction
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )
        
    @property
    def prior_weight(self) -> torch.Tensor:
        """Effective prior weight with softplus floor at 0.3."""
        return self.min_prior_weight + F.softplus(self._prior_weight)
    
    def predict_physics(self, state: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Predict object motion using prior + learned correction."""
        batch_size = state.shape[0]
        
        # Prior prediction (gravity)
        prior = self.gravity_prior.unsqueeze(0).expand(batch_size, -1).to(state.device)
        
        # Learned correction
        correction = self.correction_net(state)
        
        # Blend
        w = self.prior_weight
        prediction = w * prior + (1 - w) * correction
        
        return prediction, {'prior_weight': w.item()}
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass: state → action."""
        physics_pred, _ = self.predict_physics(state)
        combined = torch.cat([state, physics_pred], dim=-1)
        return self.policy(combined)


class BaselinePolicy(nn.Module):
    """Standard MLP policy without physics priors."""
    
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.policy = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.policy(state)


class SimulatedMetaWorld:
    """
    Simulated Meta-World environment for testing without full installation.
    
    Mimics pick-place task dynamics for validation.
    """
    
    def __init__(self, task: str = "pick-place-v2"):
        self.task = task
        self.state_dim = 39  # Meta-World observation dim
        self.action_dim = 4  # Gripper position + grasp
        
        # Simulated state
        self.gripper_pos = np.array([0.0, 0.0, 0.2])
        self.object_pos = np.array([0.1, 0.0, 0.02])
        self.goal_pos = np.array([0.0, 0.2, 0.02])
        self.holding = False
        self.step_count = 0
        self.max_steps = 150
        
    def reset(self) -> np.ndarray:
        """Reset environment."""
        self.gripper_pos = np.array([0.0, 0.0, 0.2])
        self.object_pos = np.array([0.1, 0.0, 0.02]) + np.random.randn(3) * 0.02
        self.goal_pos = np.array([0.0, 0.2, 0.02])
        self.holding = False
        self.step_count = 0
        return self._get_obs()
    
    def _get_obs(self) -> np.ndarray:
        """Get observation vector."""
        obs = np.zeros(self.state_dim)
        obs[0:3] = self.gripper_pos
        obs[3:6] = self.object_pos
        obs[6:9] = self.goal_pos
        obs[9] = float(self.holding)
        # Fill rest with zeros (placeholders for velocities, etc.)
        return obs.astype(np.float32)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Take action, return (obs, reward, terminated, truncated, info)."""
        self.step_count += 1
        
        # Move gripper
        self.gripper_pos += action[:3] * 0.02
        self.gripper_pos = np.clip(self.gripper_pos, -0.5, 0.5)
        
        # Grasp logic
        grasp_command = action[3]
        dist_to_object = np.linalg.norm(self.gripper_pos - self.object_pos)
        
        if grasp_command > 0 and dist_to_object < 0.05:
            self.holding = True
        elif grasp_command < -0.5:
            self.holding = False
        
        # Object follows gripper if holding, otherwise falls (gravity)
        if self.holding:
            self.object_pos = self.gripper_pos.copy()
        else:
            # Gravity effect
            self.object_pos[2] = max(0.02, self.object_pos[2] - 0.01)
        
        # Reward: distance to goal
        dist_to_goal = np.linalg.norm(self.object_pos - self.goal_pos)
        reward = -dist_to_goal
        
        # Success: object at goal
        success = dist_to_goal < 0.05
        if success:
            reward += 10.0
        
        terminated = success
        truncated = self.step_count >= self.max_steps
        
        return self._get_obs(), reward, terminated, truncated, {'success': success}
    
    @property
    def observation_space(self):
        return type('Space', (), {'shape': (self.state_dim,)})()
    
    @property 
    def action_space(self):
        return type('Space', (), {'shape': (self.action_dim,)})()


def create_environment(task: str) -> object:
    """Create environment (real or simulated)."""
    if HAS_METAWORLD:
        try:
            # Try v3 naming first
            task_v3 = task if task.endswith('-v3') else task.replace('-v2', '-v3')
            ml1 = metaworld.ML1(task_v3)
            env = ml1.train_classes[task_v3]()
            task_obj = ml1.train_tasks[0]  # Get first task
            env.set_task(task_obj)
            print(f"Created MetaWorld environment: {task_v3}")
            return env
        except Exception as e:
            print(f"MetaWorld creation failed: {e}. Using simulation.")
    
    return SimulatedMetaWorld(task)


def collect_demos(env, n_demos: int) -> List[Tuple]:
    """Collect demonstration trajectories (scripted policy)."""
    demos = []
    
    for _ in range(n_demos):
        reset_result = env.reset()
        # Handle both old (obs) and new (obs, info) API
        obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        trajectory = []
        
        for step in range(150):
            # Simple scripted policy: move toward object, grasp, move to goal
            if hasattr(env, 'object_pos'):
                # Simulated env
                if not env.holding:
                    # Move to object
                    direction = env.object_pos - env.gripper_pos
                    action = np.zeros(4)
                    action[:3] = direction * 2.0
                    action[3] = 1.0 if np.linalg.norm(direction) < 0.05 else 0.0
                else:
                    # Move to goal
                    direction = env.goal_pos - env.gripper_pos
                    action = np.zeros(4)
                    action[:3] = direction * 2.0
                    action[3] = 0.5
            else:
                # Real MetaWorld: use env._target_pos for goal
                gripper_pos = obs[0:3]
                
                # Get goal from environment (not observation)
                if hasattr(env, '_target_pos'):
                    goal_pos = env._target_pos
                else:
                    goal_pos = obs[36:39]  # Fallback
                
                # For pick-place, also need object position
                object_pos = obs[4:7]
                
                # Strategy depends on task
                if 'pick' in getattr(env, 'env_name', '') or 'pick' in str(type(env)):
                    # Pick-place: go to object, pick up, go to goal
                    dist_to_obj = np.linalg.norm(gripper_pos - object_pos)
                    if dist_to_obj > 0.05:
                        direction = (object_pos - gripper_pos) * 10
                        action = np.concatenate([np.clip(direction, -1, 1), [-1.0]])
                    else:
                        direction = (goal_pos - gripper_pos) * 10
                        action = np.concatenate([np.clip(direction, -1, 1), [1.0]])
                else:
                    # Reach: just go to goal
                    direction = (goal_pos - gripper_pos) * 10
                    action = np.concatenate([np.clip(direction, -1, 1), [0.0]])
            
            action = np.clip(action, -1, 1).astype(np.float32)
            step_result = env.step(action)
            # Handle both 4-tuple and 5-tuple returns
            if len(step_result) == 5:
                next_obs, reward, terminated, truncated, info = step_result
            else:
                next_obs, reward, done, info = step_result
                terminated = done
                truncated = False
            
            trajectory.append((obs, action, reward, next_obs, info.get('success', False)))
            obs = next_obs
            
            if terminated or truncated:
                break
        
        demos.append(trajectory)
    
    return demos


def train_policy(
    policy: nn.Module,
    env,
    demos: List[Tuple],
    n_steps: int,
    device: str = 'cpu'
) -> Dict[str, List[float]]:
    """Train policy using behavior cloning + online fine-tuning."""
    policy = policy.to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    
    history = {'loss': [], 'success_rate': []}
    
    # Flatten demos
    demo_states = []
    demo_actions = []
    for traj in demos:
        for obs, action, _, _, _ in traj:
            demo_states.append(obs)
            demo_actions.append(action)
    
    demo_states = torch.tensor(np.array(demo_states), dtype=torch.float32, device=device)
    demo_actions = torch.tensor(np.array(demo_actions), dtype=torch.float32, device=device)
    
    # Training loop
    for step in range(n_steps):
        # Behavior cloning on demos
        idx = torch.randint(0, len(demo_states), (32,))
        batch_states = demo_states[idx]
        batch_actions = demo_actions[idx]
        
        pred_actions = policy(batch_states)
        loss = F.mse_loss(pred_actions, batch_actions)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        history['loss'].append(loss.item())
        
        # Evaluate periodically
        if (step + 1) % 500 == 0:
            success_rate = evaluate_policy(policy, env, n_episodes=10, device=device)
            history['success_rate'].append(success_rate)
            print(f"  Step {step+1}: loss={loss.item():.4f}, success={success_rate:.2%}")
    
    return history


def evaluate_policy(
    policy: nn.Module,
    env,
    n_episodes: int,
    device: str = 'cpu'
) -> float:
    """Evaluate policy success rate."""
    policy.eval()
    successes = 0
    
    with torch.no_grad():
        for _ in range(n_episodes):
            reset_result = env.reset()
            obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
            
            for _ in range(150):
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                action = policy(obs_tensor).squeeze(0).cpu().numpy()
                
                step_result = env.step(action)
                if len(step_result) == 5:
                    obs, _, terminated, truncated, info = step_result
                else:
                    obs, _, done, info = step_result
                    terminated = done
                    truncated = False
                
                # Check success (handle both float and bool)
                success = info.get('success', 0)
                if success == 1.0 or success is True:
                    successes += 1
                    break
                
                if terminated or truncated:
                    break
    
    policy.train()
    return successes / n_episodes


def run_condition(
    prior_weight: float,
    env,
    demos: List[Tuple],
    config: LethalityConfig,
    seed: int,
    device: str
) -> Dict:
    """Run single experimental condition."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    if prior_weight == 0.0:
        # Baseline: no physics prior
        policy = BaselinePolicy(state_dim, action_dim)
        condition_name = "baseline"
    else:
        policy = AdaptivePhysicsPriorSimple(state_dim, action_dim, prior_weight)
        condition_name = f"prior_{prior_weight}"
    
    print(f"  Training {condition_name} (seed={seed})...")
    history = train_policy(policy, env, demos, config.training_steps, device)
    
    # Final evaluation
    final_success = evaluate_policy(policy, env, config.eval_episodes, device)
    
    # Get final prior weight if applicable
    final_prior_weight = None
    if hasattr(policy, 'prior_weight'):
        final_prior_weight = policy.prior_weight.item()
    
    return {
        'condition': condition_name,
        'prior_weight_init': prior_weight,
        'prior_weight_final': final_prior_weight,
        'seed': seed,
        'final_success_rate': final_success,
        'history': history
    }


def run_lethality_test(config: LethalityConfig, device: str = 'cpu') -> Dict:
    """Run the full lethality test."""
    print("=" * 60)
    print("NSCA LETHALITY TEST")
    print("=" * 60)
    print(f"Task: {config.task}")
    print(f"Demos: {config.n_demos}")
    print(f"Seeds: {config.n_seeds}")
    print(f"Conditions: {config.conditions}")
    print("=" * 60)
    
    # Create environment
    env = create_environment(config.task)
    print(f"Environment: {type(env).__name__}")
    
    # Collect demonstrations (same for all conditions)
    print(f"\nCollecting {config.n_demos} demonstrations...")
    demos = collect_demos(env, config.n_demos)
    
    # Run all conditions
    results = []
    
    for prior_weight in config.conditions:
        print(f"\n--- Condition: prior_weight={prior_weight} ---")
        
        for seed in range(config.n_seeds):
            result = run_condition(prior_weight, env, demos, config, seed, device)
            results.append(result)
    
    # Aggregate results
    summary = aggregate_results(results, config.conditions)
    
    # Print summary
    print_summary(summary)
    
    # Save results
    os.makedirs(config.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(config.output_dir, f"lethality_test_{timestamp}.json")
    
    with open(output_file, 'w') as f:
        json.dump({
            'config': config.__dict__,
            'results': results,
            'summary': summary
        }, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_file}")
    
    return summary


def aggregate_results(results: List[Dict], conditions: Tuple[float, ...]) -> Dict:
    """Aggregate results by condition."""
    summary = {}
    
    for cond in conditions:
        cond_results = [r for r in results if r['prior_weight_init'] == cond]
        success_rates = [r['final_success_rate'] for r in cond_results]
        
        summary[f"prior_{cond}"] = {
            'mean': np.mean(success_rates),
            'std': np.std(success_rates),
            'min': np.min(success_rates),
            'max': np.max(success_rates),
            'n': len(success_rates)
        }
    
    # Compute effect size (Cohen's d) between 0.9 and 0.5
    if 0.9 in conditions and 0.5 in conditions:
        high_prior = [r['final_success_rate'] for r in results if r['prior_weight_init'] == 0.9]
        low_prior = [r['final_success_rate'] for r in results if r['prior_weight_init'] == 0.5]
        
        pooled_std = np.sqrt((np.var(high_prior) + np.var(low_prior)) / 2)
        if pooled_std > 0:
            cohens_d = (np.mean(high_prior) - np.mean(low_prior)) / pooled_std
        else:
            cohens_d = 0
        
        summary['cohens_d'] = cohens_d
        summary['difference'] = np.mean(high_prior) - np.mean(low_prior)
    
    return summary


def print_summary(summary: Dict):
    """Print formatted summary."""
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    print(f"\n{'Condition':<20} {'Mean':<10} {'Std':<10} {'Range'}")
    print("-" * 60)
    
    for key, stats in summary.items():
        if key.startswith('prior_'):
            print(f"{key:<20} {stats['mean']:.2%}    {stats['std']:.2%}    [{stats['min']:.2%}, {stats['max']:.2%}]")
    
    if 'cohens_d' in summary:
        print(f"\nEffect size (Cohen's d): {summary['cohens_d']:.2f}")
        print(f"Difference (0.9 - 0.5): {summary['difference']:.2%}")
    
    # Verdict
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    
    if 'difference' in summary:
        diff = summary['difference']
        if diff > 0.20:
            print("✅ PASS: prior_weight=0.9 beats 0.5 by >20%")
            print("   → Core hypothesis validated. Proceed to full training.")
        elif diff > 0.10:
            print("⚠️  MARGINAL: prior_weight=0.9 beats 0.5 by 10-20%")
            print("   → Effect exists but weak. Consider debugging prior mechanism.")
        elif diff > 0:
            print("⚠️  WEAK: prior_weight=0.9 beats 0.5 by <10%")
            print("   → Effect may be noise. Run more seeds or debug.")
        else:
            print("❌ FAIL: prior_weight=0.9 does NOT beat 0.5")
            print("   → Prior mechanism may be broken. Debug before proceeding.")
    
    # Check if anti-prior (0.1) beats others (very bad)
    if 'prior_0.1' in summary and 'prior_0.9' in summary:
        if summary['prior_0.1']['mean'] > summary['prior_0.9']['mean']:
            print("\n🚨 CRITICAL: Anti-prior (0.1) beats strong prior (0.9)!")
            print("   → Your 'priors' are HURTING performance. Redesign architecture.")


def main():
    parser = argparse.ArgumentParser(description="NSCA Lethality Test")
    parser.add_argument("--task", default="pick-place-v2", help="Meta-World task")
    parser.add_argument("--n-seeds", type=int, default=5, help="Number of random seeds")
    parser.add_argument("--demos", type=int, default=5, help="Number of demonstrations")
    parser.add_argument("--steps", type=int, default=5000, help="Training steps")
    parser.add_argument("--device", default="cpu", help="Device (cpu/cuda/mps)")
    parser.add_argument("--output-dir", default="results/lethality_test", help="Output directory")
    
    args = parser.parse_args()
    
    config = LethalityConfig(
        task=args.task,
        n_seeds=args.n_seeds,
        n_demos=args.demos,
        training_steps=args.steps,
        output_dir=args.output_dir
    )
    
    run_lethality_test(config, device=args.device)


if __name__ == "__main__":
    main()
