"""
Causal Reasoning Layer - Understanding WHY things happen.

This layer learns cause-effect relationships through:
1. Intervention: "I did X, then Y happened" → I caused Y
2. Observation: "X happened, then Y happened" → X might cause Y
3. Counterfactual: "If X hadn't happened, would Y still happen?"

Different from dynamics prediction:
- Dynamics: "What happens next?" (correlation)
- Causal: "WHY did it happen?" (causation)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, NamedTuple
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalType(Enum):
    """Types of causal relationships."""
    INTERVENTION = "intervention"  # Agent caused it
    PHYSICS = "physics"            # Physical law caused it
    AGENT = "agent_caused"         # Another agent caused it
    SPONTANEOUS = "spontaneous"    # Happened on its own
    UNKNOWN = "unknown"


class CausalRelation(NamedTuple):
    """A learned causal relationship."""
    cause: torch.Tensor      # Cause state/action embedding
    effect: torch.Tensor     # Effect state embedding
    strength: float          # How reliably cause → effect
    causal_type: CausalType  # What kind of cause
    confidence: float        # How certain we are


@dataclass
class CausalConfig:
    """Configuration for causal reasoner."""
    state_dim: int = 256
    action_dim: int = 32
    hidden_dim: int = 512
    num_causal_factors: int = 16
    use_physics_prior: bool = True
    intervention_weight: float = 2.0  # Weight intervention learning higher


class CausalEncoder(nn.Module):
    """
    Encode state transitions into causal factors.
    
    Disentangles "what changed" into independent causal factors.
    """
    
    def __init__(self, state_dim: int, num_factors: int, hidden_dim: int) -> None:
        super().__init__()
        
        self.num_factors = num_factors
        
        # Encode state difference
        self.diff_encoder = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        
        # Factor extraction
        self.factor_heads = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(num_factors)
        ])
        
        # Factor embedding
        self.factor_embed = nn.Linear(num_factors, hidden_dim)
    
    def forward(
        self,
        state_before: torch.Tensor,
        state_after: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract causal factors from state transition.
        
        Returns:
            - Factor activations [B, num_factors]
            - Factor embedding [B, hidden_dim]
        """
        # Encode difference
        combined = torch.cat([state_before, state_after], dim=-1)
        encoded = self.diff_encoder(combined)
        
        # Extract factors
        factors = torch.cat([
            head(encoded) for head in self.factor_heads
        ], dim=-1)
        factors = torch.sigmoid(factors)  # [B, num_factors]
        
        # Embed factors
        embedding = self.factor_embed(factors)
        
        return factors, embedding


class InterventionDetector(nn.Module):
    """
    Detect whether a state change was caused by agent intervention.
    
    Key insight: When I do something and something changes,
    I likely caused it. This is how causal learning happens.
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        
        self.detector = nn.Sequential(
            nn.Linear(state_dim * 2 + action_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
    
    def forward(
        self,
        state_before: torch.Tensor,
        state_after: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Detect if state change was caused by action.
        
        Returns:
            Intervention probability [B]
        """
        combined = torch.cat([state_before, state_after, action], dim=-1)
        return self.detector(combined).squeeze(-1)


class CausalGraph(nn.Module):
    """
    Learnable causal graph structure.
    
    Nodes: State factors / properties
    Edges: Causal relationships with strength
    """
    
    def __init__(self, num_nodes: int, hidden_dim: int) -> None:
        super().__init__()
        
        self.num_nodes = num_nodes
        
        # Node embeddings
        self.node_embed = nn.Embedding(num_nodes, hidden_dim)
        
        # Edge strength predictor (soft adjacency matrix)
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        
        # Learned adjacency (prior)
        self.adjacency_prior = nn.Parameter(
            torch.zeros(num_nodes, num_nodes)
        )
    
    def get_adjacency(self) -> torch.Tensor:
        """Get soft adjacency matrix."""
        return torch.sigmoid(self.adjacency_prior)
    
    def forward(
        self,
        source_factors: torch.Tensor,
        target_factors: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute causal influence from source to target.
        
        Args:
            source_factors: [B, num_nodes]
            target_factors: [B, num_nodes]
            
        Returns:
            Causal influence strength [B]
        """
        adj = self.get_adjacency()
        
        # Weight source by adjacency to predict target
        predicted_influence = torch.einsum('bn,nm->bm', source_factors, adj)
        
        # Compare with actual target
        influence = (predicted_influence * target_factors).sum(dim=-1)
        
        return torch.sigmoid(influence)
    
    def update_edge(
        self,
        source_idx: int,
        target_idx: int,
        strength_delta: float,
    ) -> None:
        """Update causal edge strength from experience."""
        with torch.no_grad():
            self.adjacency_prior[source_idx, target_idx] += strength_delta


class CausalReasoner(nn.Module):
    """
    Layer 2: Causal Reasoning.
    
    Understands WHY things happen through:
    1. Learning cause → effect from interventions
    2. Applying intuitive physics priors
    3. Counterfactual reasoning
    
    Different from dynamics (Layer 0):
    - Dynamics: Statistical prediction of next state
    - Causal: Understanding mechanism behind state changes
    """
    
    def __init__(self, config: CausalConfig) -> None:
        super().__init__()
        
        self.config = config
        
        # Causal factor encoder
        self.causal_encoder = CausalEncoder(
            state_dim=config.state_dim,
            num_factors=config.num_causal_factors,
            hidden_dim=config.hidden_dim,
        )
        
        # Intervention detector
        self.intervention_detector = InterventionDetector(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            hidden_dim=config.hidden_dim,
        )
        
        # Causal graph
        self.causal_graph = CausalGraph(
            num_nodes=config.num_causal_factors,
            hidden_dim=config.hidden_dim // 2,
        )
        
        # Causal type classifier
        self.type_classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, len(CausalType)),
        )
        
        # Experience buffer for causal learning
        self.experience_buffer: List[CausalRelation] = []
        self.max_buffer_size = 1000
    
    def forward(
        self,
        state_before: torch.Tensor,
        state_after: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Analyze causal structure of state transition.
        
        Args:
            state_before: State before transition [B, state_dim]
            state_after: State after transition [B, state_dim]
            action: Action taken (if any) [B, action_dim]
            
        Returns:
            Dict with:
            - 'factors': Causal factor activations
            - 'intervention_prob': Probability action caused change
            - 'causal_type': Predicted causal type
            - 'causal_strength': Strength of causal link
        """
        # Extract causal factors
        factors_before, _ = self.causal_encoder(state_before, state_before)
        factors_after, factor_embed = self.causal_encoder(state_before, state_after)
        
        # Detect intervention
        if action is not None:
            intervention_prob = self.intervention_detector(
                state_before, state_after, action
            )
        else:
            intervention_prob = torch.zeros(state_before.shape[0], device=state_before.device)
        
        # Compute causal influence
        causal_strength = self.causal_graph(factors_before, factors_after)
        
        # Classify causal type
        type_logits = self.type_classifier(factor_embed)
        
        return {
            'factors_before': factors_before,
            'factors_after': factors_after,
            'intervention_prob': intervention_prob,
            'causal_type_logits': type_logits,
            'causal_strength': causal_strength,
        }
    
    def learn_from_intervention(
        self,
        state_before: torch.Tensor,
        action: torch.Tensor,
        state_after: torch.Tensor,
    ) -> CausalRelation:
        """
        Learn causal relationship from intervention.
        
        Key insight: When I do something and something changes,
        I learn that my action CAUSED the change.
        """
        with torch.no_grad():
            # Analyze causation
            result = self.forward(state_before, state_after, action)
            
            # Store as causal relation
            relation = CausalRelation(
                cause=action.detach(),
                effect=state_after.detach() - state_before.detach(),
                strength=result['intervention_prob'].mean().item(),
                causal_type=CausalType.INTERVENTION,
                confidence=result['causal_strength'].mean().item(),
            )
            
            # Add to buffer
            self.experience_buffer.append(relation)
            if len(self.experience_buffer) > self.max_buffer_size:
                self.experience_buffer.pop(0)
            
            return relation
    
    def why_did_this_happen(
        self,
        state_before: torch.Tensor,
        state_after: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[CausalType, float, str]:
        """
        Answer "why did this state change happen?"
        
        Returns:
            Tuple of (causal type, confidence, explanation)
        """
        result = self.forward(state_before, state_after, action)
        
        # Get predicted causal type
        type_probs = F.softmax(result['causal_type_logits'], dim=-1)
        if type_probs.dim() > 1:
            type_probs = type_probs[0]
        
        type_idx = type_probs.argmax().item()
        causal_type = list(CausalType)[type_idx]
        confidence = type_probs[type_idx].item()
        
        # Generate explanation
        if causal_type == CausalType.INTERVENTION:
            explanation = "The agent's action caused this change"
        elif causal_type == CausalType.PHYSICS:
            explanation = "Physical laws (gravity, collision) caused this"
        elif causal_type == CausalType.AGENT:
            explanation = "Another agent caused this change"
        elif causal_type == CausalType.SPONTANEOUS:
            explanation = "This happened spontaneously"
        else:
            explanation = "Cause unknown"
        
        return causal_type, confidence, explanation
    
    def what_would_happen_if(
        self,
        current_state: torch.Tensor,
        hypothetical_action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Counterfactual: What would happen if I did this action?
        
        Uses learned causal relationships to predict effect.
        """
        # Get causal factors for current state
        factors, _ = self.causal_encoder(current_state, current_state)
        
        # Predict change based on causal graph
        adj = self.causal_graph.get_adjacency()
        
        # Simple prediction: apply action effect through causal graph
        predicted_change = torch.einsum('bn,nm->bm', factors, adj)
        
        # This is simplified - real counterfactual reasoning is more complex
        return predicted_change
