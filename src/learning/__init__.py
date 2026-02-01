"""
Learning module for few-shot and meta-learning.

Provides mechanisms for rapid adaptation to new concepts
without requiring thousands of examples.

Includes:
- Meta-learning (MAML-style few-shot)
- Curriculum babbling for grounded concept learning
- Elastic Weight Consolidation for continual learning
"""

from .meta_learner import MetaLearner, functional_forward
from .curriculum_babbling import (
    CurriculumBabbling,
    BabblingConfig,
    BabblingEnvironment,
    SimulatedBabblingEnvironment,
    InteractionRecord,
    run_babbling_phase,
)
from .ewc import (
    ElasticWeightConsolidation,
    EWCConfig,
    ProgressiveEWC,
    MemoryAwareEWC,
    create_ewc_for_nsca,
)

__all__ = [
    # Meta-learning
    "MetaLearner",
    "functional_forward",
    # Curriculum babbling
    "CurriculumBabbling",
    "BabblingConfig",
    "BabblingEnvironment",
    "SimulatedBabblingEnvironment",
    "InteractionRecord",
    "run_babbling_phase",
    # Continual learning (EWC)
    "ElasticWeightConsolidation",
    "EWCConfig",
    "ProgressiveEWC",
    "MemoryAwareEWC",
    "create_ewc_for_nsca",
]
