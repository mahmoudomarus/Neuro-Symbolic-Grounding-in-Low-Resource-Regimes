"""
Curriculum Babbling - Two-phase exploration protocol for grounded learning.

Phase 1 (Random): Pure random exploration to discover affordances
Phase 2 (Competence-driven): Focus on actions that show learning progress

This mirrors infant motor babbling: random exploration -> goal-directed reaching.

References:
- Smith, L., & Thelen, E. (2003). Development as a dynamic system.
- Oudeyer, P.Y., Kaplan, F., & Hafner, V.V. (2007). Intrinsic motivation systems.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import random
import math

import torch
import torch.nn as nn
import numpy as np


@dataclass
class BabblingConfig:
    """Configuration for curriculum babbling."""
    phase1_steps: int = 1000           # Random exploration steps
    phase2_steps: int = 9000           # Competence-driven steps
    min_action_history: int = 5        # Min samples before computing learnability
    learnability_window: int = 20      # Window for measuring error reduction
    temperature: float = 1.0           # Softmax temperature for action selection
    exploration_bonus: float = 1.0     # Bonus for untried actions
    

@dataclass
class InteractionRecord:
    """Record of a single interaction."""
    object_id: str
    action: str
    sensory_feedback: Dict[str, Any]
    prediction_error: float
    timestamp: int


class CurriculumBabbling:
    """
    Two-phase babbling protocol for grounded concept learning.
    
    Protocol:
    1. Phase 1 (Random): Pure random exploration
       - Builds initial affordance coverage
       - Discovers action-effect contingencies
       
    2. Phase 2 (Competence-driven): 
       - Retry actions that showed high prediction error reduction
       - Skip actions that are unlearnable (noisy TV defense)
       
    This creates the "childhood" phase where the agent builds its
    grounding table through interaction, not hard-coded values.
    """
    
    def __init__(self, config: Optional[BabblingConfig] = None) -> None:
        self.config = config or BabblingConfig()
        self.step = 0
        self.phase = 1
        
        # Track action outcomes
        self.action_history: Dict[str, List[float]] = defaultdict(list)
        self.action_counts: Dict[str, int] = defaultdict(int)
        
        # Track object-action pairs for grounding
        self.interaction_log: List[InteractionRecord] = []
        
        # Learnability scores per action
        self.learnability_scores: Dict[str, float] = {}
        
    @property
    def total_steps(self) -> int:
        return self.config.phase1_steps + self.config.phase2_steps
    
    @property
    def is_complete(self) -> bool:
        return self.step >= self.total_steps
    
    def select_action(self, available_actions: List[str]) -> str:
        """
        Select next action based on current phase.
        
        Phase 1: Random selection
        Phase 2: Competence-weighted selection
        """
        if not available_actions:
            raise ValueError("No actions available")
            
        self.step += 1
        
        # Update phase
        if self.step <= self.config.phase1_steps:
            self.phase = 1
        else:
            self.phase = 2
        
        if self.phase == 1:
            # Phase 1: Pure random exploration
            return random.choice(available_actions)
        else:
            # Phase 2: Competence-driven selection
            return self._competence_driven_selection(available_actions)
    
    def _competence_driven_selection(self, available_actions: List[str]) -> str:
        """
        Select action based on learnability (prediction error reduction).
        
        Prefer actions that:
        1. Have shown learning progress (high learnability)
        2. Haven't been tried much (exploration bonus)
        """
        scores = []
        
        for action in available_actions:
            history = self.action_history[action]
            
            if len(history) >= self.config.min_action_history:
                # Compute learnability = error reduction over time
                learnability = self._compute_learnability(history)
                self.learnability_scores[action] = learnability
                
                # High learnability = high score (worth retrying)
                # Low learnability = low score (might be unlearnable/noisy)
                score = max(0.1, learnability)
            else:
                # Untried actions get exploration bonus
                score = self.config.exploration_bonus
            
            scores.append(score)
        
        # Softmax selection with temperature
        scores = np.array(scores)
        exp_scores = np.exp(scores / self.config.temperature)
        probs = exp_scores / np.sum(exp_scores)
        
        return np.random.choice(available_actions, p=probs)
    
    def _compute_learnability(self, history: List[float]) -> float:
        """
        Compute learnability from prediction error history.
        
        Learnability = (early_error - recent_error) / early_error
        
        High learnability: Errors decrease (learnable)
        Low learnability: Errors stay high (might be noisy TV)
        """
        if len(history) < self.config.min_action_history:
            return 1.0  # Assume learnable until proven otherwise
        
        window = min(self.config.learnability_window, len(history) // 2)
        if window < 3:
            return 1.0
            
        early_error = np.mean(history[:window])
        recent_error = np.mean(history[-window:])
        
        if early_error < 1e-6:
            return 0.0  # Already at zero error
        
        learnability = (early_error - recent_error) / (early_error + 1e-6)
        return max(0, learnability)  # Clamp to non-negative
    
    def record_interaction(
        self,
        object_id: str,
        action: str,
        sensory_feedback: Dict[str, Any],
        prediction_error: float,
    ) -> None:
        """
        Record an interaction outcome.
        
        This builds the grounding table through experience.
        """
        # Update action history
        self.action_history[action].append(prediction_error)
        self.action_counts[action] += 1
        
        # Keep history bounded
        max_history = self.config.learnability_window * 5
        if len(self.action_history[action]) > max_history:
            self.action_history[action] = self.action_history[action][-max_history:]
        
        # Log interaction
        record = InteractionRecord(
            object_id=object_id,
            action=action,
            sensory_feedback=sensory_feedback,
            prediction_error=prediction_error,
            timestamp=self.step,
        )
        self.interaction_log.append(record)
    
    def get_grounding_data(self, object_id: str) -> List[InteractionRecord]:
        """Get all interactions with a specific object."""
        return [r for r in self.interaction_log if r.object_id == object_id]
    
    def get_action_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for each action."""
        stats = {}
        
        for action in self.action_history:
            history = self.action_history[action]
            stats[action] = {
                'count': self.action_counts[action],
                'mean_error': np.mean(history) if history else 0.0,
                'std_error': np.std(history) if len(history) > 1 else 0.0,
                'learnability': self.learnability_scores.get(action, 1.0),
            }
        
        return stats
    
    def get_phase_progress(self) -> Dict[str, Any]:
        """Get current progress information."""
        return {
            'phase': self.phase,
            'step': self.step,
            'total_steps': self.total_steps,
            'progress': self.step / self.total_steps,
            'phase1_complete': self.step > self.config.phase1_steps,
            'is_complete': self.is_complete,
            'unique_actions': len(self.action_history),
            'total_interactions': len(self.interaction_log),
        }


class BabblingEnvironment:
    """
    Abstract interface for babbling environments.
    
    Implementations should provide:
    - Available actions
    - Sensory feedback from interactions
    - Object identification
    """
    
    def get_available_actions(self) -> List[str]:
        """Return list of available actions."""
        raise NotImplementedError
    
    def execute_action(self, action: str) -> Dict[str, Any]:
        """
        Execute action and return sensory feedback.
        
        Returns dict with keys like:
        - 'audio': Audio features from action
        - 'visual': Visual features after action
        - 'force': Force feedback (if available)
        - 'acceleration': Object acceleration
        """
        raise NotImplementedError
    
    def get_current_object_id(self) -> str:
        """Return identifier for current object."""
        raise NotImplementedError


class SimulatedBabblingEnvironment(BabblingEnvironment):
    """
    Simulated environment for testing babbling protocol.
    
    Provides realistic-ish sensory feedback for common actions.
    
    IMPLEMENTATION NOTE (Cold Start Trap):
    Objects are always initialized "within reach" of the agent.
    If the probability of interaction is too low (<1%), the agent
    will never learn action-feedback correlations.
    
    Set A (Training): wooden blocks, plastic toys, rubber balls
    Set B (Evaluation): ceramics, foams, metals (never seen during babbling)
    """
    
    ACTIONS = ['strike', 'push', 'lift', 'squeeze', 'drop', 'shake']
    
    # Set A: Training objects (used during babbling)
    OBJECT_PROPERTIES_SET_A = {
        'wooden_block': {'hardness': 0.7, 'weight': 0.5, 'size': 0.4},
        'plastic_toy': {'hardness': 0.5, 'weight': 0.3, 'size': 0.3},
        'rubber_ball': {'hardness': 0.4, 'weight': 0.3, 'size': 0.3},
        'cloth_piece': {'hardness': 0.1, 'weight': 0.1, 'size': 0.5},
        'cardboard_box': {'hardness': 0.3, 'weight': 0.2, 'size': 0.6},
    }
    
    # Set B: Evaluation objects (novel, never seen during babbling)
    OBJECT_PROPERTIES_SET_B = {
        'ceramic_cup': {'hardness': 0.85, 'weight': 0.4, 'size': 0.3},
        'foam_block': {'hardness': 0.15, 'weight': 0.05, 'size': 0.5},
        'metal_cube': {'hardness': 1.0, 'weight': 0.9, 'size': 0.3},
        'glass_bottle': {'hardness': 0.9, 'weight': 0.5, 'size': 0.5},
        'gel_sphere': {'hardness': 0.2, 'weight': 0.2, 'size': 0.3},
    }
    
    # Legacy alias for backward compatibility
    OBJECT_PROPERTIES = OBJECT_PROPERTIES_SET_A
    
    # Spawn modes for forced interaction initialization
    # This prevents the "sparse reward problem" where random exploration
    # rarely contacts objects (99% airballs)
    SPAWN_MODES = {
        'drop': 'gripper_holding_object',       # Gripper starts holding object
        'strike': 'gripper_inches_from_object', # Gripper near object face
        'push': 'gripper_touching_object',      # Gripper in contact
        'lift': 'gripper_at_object',            # Gripper at grasp position
        'squeeze': 'gripper_holding_object',    # Gripper holding
        'shake': 'gripper_holding_object',      # Gripper holding
    }
    
    def __init__(
        self, 
        noise_level: float = 0.1,
        use_set: str = 'A',  # 'A' for training, 'B' for evaluation
        interaction_probability: float = 1.0,  # Objects always within reach
        forced_interaction: bool = True,  # Use spawn modes for guaranteed contact
    ):
        self.noise_level = noise_level
        self.interaction_probability = interaction_probability
        self.forced_interaction = forced_interaction
        
        # Select object set
        if use_set == 'B':
            self.active_objects = self.OBJECT_PROPERTIES_SET_B
        else:
            self.active_objects = self.OBJECT_PROPERTIES_SET_A
        
        self.current_object = random.choice(list(self.active_objects.keys()))
        
        # State for forced interaction
        self._spawn_mode = None
        self._holding_object = False
    
    def get_available_actions(self) -> List[str]:
        return self.ACTIONS
    
    def get_current_object_id(self) -> str:
        return self.current_object
    
    def set_object(self, object_id: str) -> None:
        if object_id in self.active_objects:
            self.current_object = object_id
        else:
            self.current_object = random.choice(list(self.active_objects.keys()))
    
    def randomize_object(self) -> str:
        """Randomly select a new object from the active set."""
        self.current_object = random.choice(list(self.active_objects.keys()))
        return self.current_object
    
    def setup_for_action(self, action: str) -> None:
        """
        Setup environment state for forced interaction.
        
        This implements the "Forced Interaction Initialization" pattern
        to avoid the sparse reward problem.
        """
        if not self.forced_interaction:
            return
        
        spawn_mode = self.SPAWN_MODES.get(action, 'gripper_at_object')
        self._spawn_mode = spawn_mode
        
        if spawn_mode == 'gripper_holding_object':
            self._holding_object = True
        elif spawn_mode in ['gripper_inches_from_object', 'gripper_touching_object', 'gripper_at_object']:
            self._holding_object = False
    
    def execute_action(self, action: str) -> Dict[str, Any]:
        """Generate sensory feedback based on action and object."""
        # Setup forced interaction if enabled
        self.setup_for_action(action)
        
        # Cold start protection: objects are always within reach
        # (interaction_probability = 1.0 by default)
        if random.random() > self.interaction_probability:
            # Object out of reach - no feedback
            return {'no_contact': True, 'action': action}
        
        props = self.active_objects[self.current_object]
        noise = lambda: random.gauss(0, self.noise_level)
        
        feedback = {}
        
        if action == 'strike':
            # Hardness correlates with audio frequency
            feedback['audio_frequency'] = props['hardness'] + noise()
            feedback['audio_intensity'] = 0.5 + noise()
            
        elif action == 'push':
            # Weight correlates with resistance to push
            feedback['displacement'] = (1 - props['weight']) * 0.5 + noise()
            feedback['resistance'] = props['weight'] + noise()
            
        elif action == 'lift':
            # Direct weight sensing
            feedback['force_required'] = props['weight'] + noise()
            feedback['acceleration'] = (1 - props['weight']) + noise()
            
        elif action == 'squeeze':
            # Hardness from deformation
            feedback['deformation'] = (1 - props['hardness']) + noise()
            feedback['resistance'] = props['hardness'] + noise()
            
        elif action == 'drop':
            # Sound from impact
            feedback['impact_sound'] = props['hardness'] * props['weight'] + noise()
            feedback['bounce'] = (1 - props['weight']) * 0.3 + noise()
            
        elif action == 'shake':
            # Infer from momentum
            feedback['inertia'] = props['weight'] + noise()
            feedback['sound'] = random.random() * 0.3 + noise()
        
        # Clamp all values to [0, 1]
        feedback = {k: max(0, min(1, v)) for k, v in feedback.items()}
        
        return feedback


def run_babbling_phase(
    babbling: CurriculumBabbling,
    env: BabblingEnvironment,
    predictor: Optional[nn.Module] = None,
) -> Dict[str, Any]:
    """
    Run complete babbling phase.
    
    Args:
        babbling: Curriculum babbling controller
        env: Environment to interact with
        predictor: Optional model for computing prediction errors
        
    Returns:
        Statistics from babbling phase
    """
    while not babbling.is_complete:
        # Get available actions
        actions = env.get_available_actions()
        
        # Select action
        action = babbling.select_action(actions)
        
        # Execute and get feedback
        feedback = env.execute_action(action)
        object_id = env.get_current_object_id()
        
        # Compute prediction error (simplified)
        if predictor is not None:
            # Use actual predictor
            pred_error = 0.5  # Placeholder
        else:
            # Random baseline error that decreases over time
            base_error = 0.8
            decay = babbling.step / babbling.total_steps
            pred_error = base_error * (1 - decay * 0.5) + random.gauss(0, 0.1)
            pred_error = max(0, min(1, pred_error))
        
        # Record interaction
        babbling.record_interaction(
            object_id=object_id,
            action=action,
            sensory_feedback=feedback,
            prediction_error=pred_error,
        )
    
    return {
        'progress': babbling.get_phase_progress(),
        'action_stats': babbling.get_action_statistics(),
        'total_interactions': len(babbling.interaction_log),
    }
