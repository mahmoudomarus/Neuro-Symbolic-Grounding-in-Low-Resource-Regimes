"""
Elastic Weight Consolidation (EWC) for Continual Learning.

Prevents catastrophic forgetting by protecting important weights
learned on previous tasks.

Key insight from peer review: Having dual memory (episodic/semantic)
doesn't prevent the neural network weights from being overwritten.
EWC solves this by adding a penalty for changing important weights.

Integration with NSCA:
- Semantic memory parameters get 10x protection (consolidated knowledge)
- Episodic memory parameters get 1x protection (can be overwritten)

References:
- Kirkpatrick et al. (2017). Overcoming catastrophic forgetting.
- Lopez-Paz & Ranzato (2017). Gradient Episodic Memory.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


@dataclass
class EWCConfig:
    """Configuration for EWC."""
    importance_weight: float = 1000.0  # Lambda in the EWC loss
    semantic_multiplier: float = 10.0  # Extra protection for semantic memory
    episodic_multiplier: float = 1.0   # Standard protection for episodic
    online_ewc: bool = True            # Use online EWC (more memory efficient)
    gamma: float = 0.9                 # Decay for online EWC
    num_samples: int = 200             # Samples for Fisher computation


class ElasticWeightConsolidation:
    """
    Elastic Weight Consolidation for preventing catastrophic forgetting.
    
    The core idea: After training on Task A, compute which weights were
    most important (Fisher information). When training on Task B, add
    a penalty for changing those important weights.
    
    EWC Loss = Task_B_Loss + λ * Σ_i F_i * (θ_i - θ*_i)²
    
    Where:
    - F_i = Fisher information for parameter i (importance)
    - θ_i = current parameter value
    - θ*_i = optimal parameter value after Task A
    - λ = importance weight (default 1000)
    
    NSCA Integration:
    - Semantic memory (slow, consolidated): 10x protection
    - Episodic memory (fast, recent): 1x protection
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[EWCConfig] = None,
    ) -> None:
        self.model = model
        self.config = config or EWCConfig()
        
        # Storage for Fisher information and optimal parameters
        self.fisher_information: Dict[str, torch.Tensor] = {}
        self.optimal_params: Dict[str, torch.Tensor] = {}
        
        # Track which parameters belong to which memory type
        self.param_memory_type: Dict[str, str] = {}
        self._categorize_parameters()
        
        # For online EWC
        self.task_count = 0
    
    def _categorize_parameters(self) -> None:
        """
        Categorize parameters into semantic vs episodic memory.
        
        Semantic: Slow-changing, consolidated knowledge
        Episodic: Fast-changing, recent experiences
        """
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            # Heuristic: Categorize based on layer name
            name_lower = name.lower()
            
            if any(kw in name_lower for kw in ['semantic', 'prototype', 'prior', 'consolidated']):
                self.param_memory_type[name] = 'semantic'
            elif any(kw in name_lower for kw in ['episodic', 'recent', 'buffer']):
                self.param_memory_type[name] = 'episodic'
            else:
                # Default: treat as semantic (more protection)
                self.param_memory_type[name] = 'semantic'
    
    def compute_fisher(
        self,
        dataloader: DataLoader,
        loss_fn: Optional[callable] = None,
        num_samples: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute Fisher information matrix (DIAGONAL approximation).
        
        IMPLEMENTATION NOTE (Fisher Explosion Trap):
        We compute only the DIAGONAL of the Fisher matrix, not the full matrix.
        For 16M parameters, the full Fisher would be 16M × 16M = 256 trillion elements.
        The diagonal approximation is standard practice and usually sufficient.
        
        Fisher information tells us how important each parameter is
        for the current task. Parameters with high Fisher = important.
        
        F_i = E[(∂L/∂θ_i)²]  (diagonal only)
        
        Args:
            dataloader: DataLoader for current task
            loss_fn: Loss function (if None, assumes model.compute_loss)
            num_samples: Number of samples for estimation
            
        Returns:
            Dict mapping parameter name to DIAGONAL Fisher information
        """
        num_samples = num_samples or self.config.num_samples
        
        self.model.eval()
        
        # DIAGONAL Fisher only - each param stores just its own importance
        fisher = {
            name: torch.zeros_like(param)
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }
        
        sample_count = 0
        for batch in dataloader:
            if sample_count >= num_samples:
                break
            
            self.model.zero_grad()
            
            # Compute loss
            if loss_fn is not None:
                loss = loss_fn(self.model, batch)
            elif hasattr(self.model, 'compute_loss'):
                loss = self.model.compute_loss(batch)
            else:
                # Default: assume batch is (input, target)
                if isinstance(batch, (tuple, list)):
                    x, y = batch[0], batch[1]
                    output = self.model(x)
                    loss = F.cross_entropy(output, y)
                else:
                    raise ValueError("Cannot compute loss without loss_fn")
            
            loss.backward()
            
            # Accumulate squared gradients (DIAGONAL Fisher)
            # This is memory-efficient: O(n) instead of O(n²)
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.detach().pow(2)
            
            sample_count += len(batch[0]) if isinstance(batch, (tuple, list)) else 1
        
        # Normalize
        for name in fisher:
            fisher[name] /= max(sample_count, 1)
        
        return fisher
    
    def consolidate(
        self,
        dataloader: DataLoader,
        loss_fn: Optional[callable] = None,
    ) -> None:
        """
        Consolidate current task knowledge.
        
        Call this after training on a task to "remember" what was learned.
        """
        # Compute Fisher for current task
        new_fisher = self.compute_fisher(dataloader, loss_fn)
        
        # Apply memory-type multipliers
        for name in new_fisher:
            memory_type = self.param_memory_type.get(name, 'semantic')
            
            if memory_type == 'semantic':
                multiplier = self.config.semantic_multiplier
            else:
                multiplier = self.config.episodic_multiplier
            
            new_fisher[name] *= multiplier
        
        if self.config.online_ewc and self.task_count > 0:
            # Online EWC: Accumulate with decay
            for name in new_fisher:
                if name in self.fisher_information:
                    self.fisher_information[name] = (
                        self.config.gamma * self.fisher_information[name] +
                        new_fisher[name]
                    )
                else:
                    self.fisher_information[name] = new_fisher[name]
        else:
            # Standard EWC: Replace
            self.fisher_information = new_fisher
        
        # Store optimal parameters
        self.optimal_params = {
            name: param.clone().detach()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }
        
        self.task_count += 1
    
    def penalty(self) -> torch.Tensor:
        """
        Compute EWC penalty to add to the loss.
        
        EWC_penalty = Σ_i F_i * (θ_i - θ*_i)²
        
        Add this to your training loop:
            loss = task_loss + ewc.penalty()
        """
        if not self.fisher_information:
            return torch.tensor(0.0, device=next(self.model.parameters()).device)
        
        penalty = torch.tensor(0.0, device=next(self.model.parameters()).device)
        
        for name, param in self.model.named_parameters():
            if name in self.fisher_information:
                fisher = self.fisher_information[name]
                optimal = self.optimal_params[name]
                
                penalty += (fisher * (param - optimal).pow(2)).sum()
        
        return self.config.importance_weight * penalty
    
    def get_importance_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics about parameter importance."""
        stats = {}
        
        for name, fisher in self.fisher_information.items():
            memory_type = self.param_memory_type.get(name, 'semantic')
            stats[name] = {
                'memory_type': memory_type,
                'mean_importance': fisher.mean().item(),
                'max_importance': fisher.max().item(),
                'total_importance': fisher.sum().item(),
            }
        
        return stats
    
    def get_protection_summary(self) -> Dict[str, Any]:
        """Get summary of EWC protection."""
        semantic_params = sum(
            1 for name in self.param_memory_type 
            if self.param_memory_type[name] == 'semantic'
        )
        episodic_params = sum(
            1 for name in self.param_memory_type
            if self.param_memory_type[name] == 'episodic'
        )
        
        return {
            'task_count': self.task_count,
            'semantic_params': semantic_params,
            'episodic_params': episodic_params,
            'semantic_multiplier': self.config.semantic_multiplier,
            'episodic_multiplier': self.config.episodic_multiplier,
            'importance_weight': self.config.importance_weight,
        }


class ProgressiveEWC(ElasticWeightConsolidation):
    """
    Progressive EWC with importance decay.
    
    Older tasks gradually become less protected, allowing the network
    to adapt to changing environments while still preserving core knowledge.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[EWCConfig] = None,
        decay_rate: float = 0.95,
    ) -> None:
        super().__init__(model, config)
        self.decay_rate = decay_rate
        self.task_fisher: List[Dict[str, torch.Tensor]] = []
    
    def consolidate(
        self,
        dataloader: DataLoader,
        loss_fn: Optional[callable] = None,
    ) -> None:
        """Consolidate with progressive decay of older tasks."""
        # Decay existing Fisher information
        for i, task_f in enumerate(self.task_fisher):
            decay = self.decay_rate ** (len(self.task_fisher) - i)
            for name in task_f:
                task_f[name] *= decay
        
        # Compute and store new Fisher
        new_fisher = self.compute_fisher(dataloader, loss_fn)
        
        # Apply memory-type multipliers
        for name in new_fisher:
            memory_type = self.param_memory_type.get(name, 'semantic')
            multiplier = (
                self.config.semantic_multiplier 
                if memory_type == 'semantic'
                else self.config.episodic_multiplier
            )
            new_fisher[name] *= multiplier
        
        self.task_fisher.append(new_fisher)
        
        # Combine all task Fisher information
        self.fisher_information = {}
        for task_f in self.task_fisher:
            for name, fisher in task_f.items():
                if name in self.fisher_information:
                    self.fisher_information[name] += fisher
                else:
                    self.fisher_information[name] = fisher.clone()
        
        # Store optimal parameters
        self.optimal_params = {
            name: param.clone().detach()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }
        
        self.task_count += 1


class MemoryAwareEWC(ElasticWeightConsolidation):
    """
    EWC with explicit integration with NSCA's dual memory system.
    
    Automatically detects semantic vs episodic parameters based on
    the NSCA architecture naming conventions.
    """
    
    SEMANTIC_KEYWORDS = [
        'semantic', 'prototype', 'prior', 'consolidated',
        'long_term', 'cortex', 'schema', 'concept',
        'property_layer', 'category', 'affordance',
    ]
    
    EPISODIC_KEYWORDS = [
        'episodic', 'recent', 'buffer', 'hippocampus',
        'short_term', 'working', 'attention',
    ]
    
    def _categorize_parameters(self) -> None:
        """Categorize parameters with NSCA-aware heuristics."""
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            name_lower = name.lower()
            
            # Check for semantic keywords
            if any(kw in name_lower for kw in self.SEMANTIC_KEYWORDS):
                self.param_memory_type[name] = 'semantic'
            # Check for episodic keywords
            elif any(kw in name_lower for kw in self.EPISODIC_KEYWORDS):
                self.param_memory_type[name] = 'episodic'
            # Special case: prior weights should be highly protected
            elif 'prior_weight' in name_lower:
                self.param_memory_type[name] = 'semantic'
            # Default to semantic (more protection is safer)
            else:
                self.param_memory_type[name] = 'semantic'


def create_ewc_for_nsca(
    model: nn.Module,
    importance_weight: float = 1000.0,
    semantic_multiplier: float = 10.0,
) -> MemoryAwareEWC:
    """
    Factory function to create EWC configured for NSCA.
    
    Args:
        model: The NSCA CognitiveAgent or similar model
        importance_weight: Base importance weight (λ)
        semantic_multiplier: Extra protection for semantic memory
        
    Returns:
        Configured MemoryAwareEWC instance
    """
    config = EWCConfig(
        importance_weight=importance_weight,
        semantic_multiplier=semantic_multiplier,
        episodic_multiplier=1.0,
        online_ewc=True,
    )
    
    return MemoryAwareEWC(model, config)
