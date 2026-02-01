"""
Meta-Learner for few-shot adaptation.

Implements Model-Agnostic Meta-Learning (MAML) style adaptation,
allowing the model to learn new concepts from just 1-5 examples.

Key insight: Instead of learning fixed weights, learn weights that
can be quickly adapted to new tasks with just a few gradient steps.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def functional_forward(
    model: nn.Module,
    x: torch.Tensor,
    params: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """
    Forward pass using custom parameters instead of model's parameters.
    
    This is needed for MAML-style meta-learning where we want to
    compute gradients through the adaptation process.
    
    Args:
        model: The neural network module
        x: Input tensor
        params: Dictionary of parameter name -> tensor
        
    Returns:
        Output tensor from forward pass with custom params
    """
    # Save original parameters
    original_params = {}
    for name, param in model.named_parameters():
        original_params[name] = param.data.clone()
        
        # Replace with custom param
        if name in params:
            param.data = params[name]
    
    # Forward pass
    try:
        output = model(x)
    finally:
        # Restore original parameters
        for name, param in model.named_parameters():
            if name in original_params:
                param.data = original_params[name]
    
    return output


class MetaLearner:
    """
    Model-Agnostic Meta-Learning for few-shot adaptation.
    
    Learn from 1-5 examples of a new concept by taking a few
    gradient steps on those examples, then evaluate on new data.
    
    Key idea: The model's weights serve as a good initialization
    that can be quickly adapted to new tasks.
    """
    
    def __init__(
        self,
        model: nn.Module,
        inner_lr: float = 0.01,
        inner_steps: int = 5,
        first_order: bool = True,
    ) -> None:
        """
        Initialize meta-learner.
        
        Args:
            model: The model to adapt (encoder, classifier, etc.)
            inner_lr: Learning rate for inner loop adaptation
            inner_steps: Number of gradient steps for adaptation
            first_order: If True, don't compute second-order derivatives
                        (faster but less accurate)
        """
        self.model = model
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.first_order = first_order
    
    def adapt(
        self,
        support_data: torch.Tensor,
        support_labels: torch.Tensor,
        loss_fn: Optional[Callable] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Adapt model to new task using support set.
        
        Args:
            support_data: Support examples [K, ...] (K examples)
            support_labels: Support labels [K]
            loss_fn: Loss function (default: contrastive loss)
            
        Returns:
            Adapted parameters dictionary
        """
        if loss_fn is None:
            loss_fn = self._default_loss
        
        # Clone current parameters
        adapted_params = OrderedDict()
        for name, param in self.model.named_parameters():
            adapted_params[name] = param.clone()
            if not self.first_order:
                adapted_params[name].requires_grad_(True)
        
        # Inner loop: few gradient steps
        for step in range(self.inner_steps):
            # Forward pass with current adapted params
            features = self._forward_with_params(support_data, adapted_params)
            
            # Compute loss
            loss = loss_fn(features, support_labels)
            
            # Compute gradients
            if self.first_order:
                # First-order approximation (faster)
                grads = torch.autograd.grad(
                    loss,
                    adapted_params.values(),
                    create_graph=False,
                )
            else:
                # Full second-order gradients
                grads = torch.autograd.grad(
                    loss,
                    adapted_params.values(),
                    create_graph=True,
                )
            
            # Update adapted parameters
            adapted_params = OrderedDict()
            for (name, param), grad in zip(
                self.model.named_parameters(), grads
            ):
                new_param = adapted_params.get(name, param.clone()) - self.inner_lr * grad
                adapted_params[name] = new_param
        
        return adapted_params
    
    def _forward_with_params(
        self,
        x: torch.Tensor,
        params: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Forward pass using adapted parameters."""
        return functional_forward(self.model, x, params)
    
    def _default_loss(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Default contrastive loss for few-shot learning.
        
        Maximizes similarity between same-class examples,
        minimizes similarity between different-class examples.
        """
        # Normalize features
        features = F.normalize(features.flatten(1), dim=1)
        
        # Compute similarity matrix
        sim_matrix = features @ features.T  # [K, K]
        
        # Create label mask
        label_matrix = labels.unsqueeze(0) == labels.unsqueeze(1)  # [K, K]
        positive_mask = label_matrix.float()
        negative_mask = (~label_matrix).float()
        
        # Temperature
        temperature = 0.1
        
        # InfoNCE loss
        exp_sim = torch.exp(sim_matrix / temperature)
        
        # For each sample, compute loss
        positive_sim = (exp_sim * positive_mask).sum(dim=1)
        all_sim = exp_sim.sum(dim=1)
        
        loss = -torch.log(positive_sim / (all_sim + 1e-8) + 1e-8).mean()
        
        return loss
    
    def evaluate(
        self,
        adapted_params: Dict[str, torch.Tensor],
        query_data: torch.Tensor,
        query_labels: torch.Tensor,
        metric: str = 'accuracy',
    ) -> float:
        """
        Evaluate adapted model on query set.
        
        Args:
            adapted_params: Parameters from adapt()
            query_data: Query examples [N, ...]
            query_labels: Query labels [N]
            metric: Evaluation metric ('accuracy', 'loss')
            
        Returns:
            Evaluation score
        """
        with torch.no_grad():
            features = self._forward_with_params(query_data, adapted_params)
            
            if metric == 'loss':
                return self._default_loss(features, query_labels).item()
            
            elif metric == 'accuracy':
                # Nearest neighbor classification using adapted features
                features = F.normalize(features.flatten(1), dim=1)
                
                # Use labels from support set as prototypes
                unique_labels = query_labels.unique()
                predictions = []
                
                for feat in features:
                    sims = []
                    for label in unique_labels:
                        mask = query_labels == label
                        class_features = features[mask]
                        class_sim = (feat @ class_features.T).mean()
                        sims.append(class_sim)
                    
                    predictions.append(unique_labels[torch.tensor(sims).argmax()])
                
                predictions = torch.stack(predictions)
                accuracy = (predictions == query_labels).float().mean().item()
                
                return accuracy
        
        return 0.0


class PrototypicalNetworks:
    """
    Prototypical Networks for few-shot learning.
    
    Simpler than MAML - computes class prototypes from support set
    and classifies query examples by nearest prototype.
    """
    
    def __init__(self, encoder: nn.Module) -> None:
        """
        Initialize prototypical networks.
        
        Args:
            encoder: Feature encoder network
        """
        self.encoder = encoder
    
    def compute_prototypes(
        self,
        support_data: torch.Tensor,
        support_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute class prototypes from support set.
        
        Args:
            support_data: Support examples [K, ...]
            support_labels: Support labels [K]
            
        Returns:
            Tuple of (prototypes [C, D], unique_labels [C])
        """
        # Encode support set
        with torch.no_grad():
            support_features = self.encoder(support_data)
            support_features = support_features.flatten(1)  # [K, D]
        
        # Compute prototype for each class
        unique_labels = support_labels.unique()
        prototypes = []
        
        for label in unique_labels:
            mask = support_labels == label
            class_features = support_features[mask]
            prototype = class_features.mean(dim=0)
            prototypes.append(prototype)
        
        prototypes = torch.stack(prototypes)  # [C, D]
        
        return prototypes, unique_labels
    
    def classify(
        self,
        query_data: torch.Tensor,
        prototypes: torch.Tensor,
        unique_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Classify query examples using prototypes.
        
        Args:
            query_data: Query examples [N, ...]
            prototypes: Class prototypes [C, D]
            unique_labels: Label for each prototype [C]
            
        Returns:
            Predicted labels [N]
        """
        # Encode query set
        with torch.no_grad():
            query_features = self.encoder(query_data)
            query_features = query_features.flatten(1)  # [N, D]
        
        # Compute distances to prototypes
        # Using negative squared Euclidean distance as similarity
        distances = torch.cdist(query_features, prototypes, p=2)  # [N, C]
        
        # Predict nearest prototype
        predictions_idx = distances.argmin(dim=1)  # [N]
        predictions = unique_labels[predictions_idx]
        
        return predictions
    
    def few_shot_classify(
        self,
        support_data: torch.Tensor,
        support_labels: torch.Tensor,
        query_data: torch.Tensor,
    ) -> torch.Tensor:
        """
        Complete few-shot classification pipeline.
        
        Args:
            support_data: Support examples [K, ...]
            support_labels: Support labels [K]
            query_data: Query examples [N, ...]
            
        Returns:
            Predicted labels for query [N]
        """
        prototypes, unique_labels = self.compute_prototypes(
            support_data, support_labels
        )
        return self.classify(query_data, prototypes, unique_labels)
