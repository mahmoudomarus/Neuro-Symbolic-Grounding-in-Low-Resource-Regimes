"""
Learning module for few-shot and meta-learning.

Provides mechanisms for rapid adaptation to new concepts
without requiring thousands of examples.
"""

from .meta_learner import MetaLearner, functional_forward

__all__ = [
    "MetaLearner",
    "functional_forward",
]
