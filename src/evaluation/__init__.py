"""
Evaluation module for NSCA benchmarks.

Provides evaluation harnesses for:
- Meta-World robotic manipulation
- Physion physics prediction
- Ablation studies (priors vs random init)
"""

from .metaworld_eval import (
    MetaWorldEvaluator,
    AblationStudy,
    EvaluationConfig,
    run_ablation_study,
    compute_effect_size,
)

__all__ = [
    "MetaWorldEvaluator",
    "AblationStudy", 
    "EvaluationConfig",
    "run_ablation_study",
    "compute_effect_size",
]
