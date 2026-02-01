"""
Motivation / Drive System (Layer 3).

Provides intrinsic motivation for learning:
- Curiosity: Seek novel, learnable experiences
- Competence: Seek mastery, reduce uncertainty
- Prediction error: Minimize surprise (or seek surprise)

Key insight: Babies learn because they NEED things.
Without drives, there's no reason to learn anything.
"""

from .drive_system import (
    DriveSystem,
    DriveConfig,
    DriveState,
)
from .intrinsic_reward import (
    IntrinsicRewardComputer,
    CuriosityReward,
    CompetenceReward,
)
from .attention import AttentionAllocator

__all__ = [
    "DriveSystem",
    "DriveConfig",
    "DriveState",
    "IntrinsicRewardComputer",
    "CuriosityReward",
    "CompetenceReward",
    "AttentionAllocator",
]
