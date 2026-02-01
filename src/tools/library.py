"""
Concrete tools for the agent: Wikipedia (information) and Calculator (math).
Trigger concepts are normalized tensors matching the encoder's latent dimension.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .base import Tool


def _normalized_concept(dim: int, seed: int) -> torch.Tensor:
    """Return a random normalized vector of shape (dim,) for use as trigger_concept."""
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(dim, generator=g)
    return F.normalize(v, dim=0)


class WikiTool(Tool):
    """
    Tool simulating an information lookup (e.g. Wikipedia).
    Trigger concept: normalized random vector simulating 'Information'.
    """

    def __init__(self, latent_dim: int, seed: int = 42) -> None:
        self._latent_dim = latent_dim
        self._concept = _normalized_concept(latent_dim, seed)

    def name(self) -> str:
        return "Wikipedia"

    def trigger_concept(self) -> torch.Tensor:
        return self._concept

    def execute(self, *args: Any) -> Any:
        return "Searching database... [Mock Result]"


class CalculatorTool(Tool):
    """
    Tool simulating a calculator (math).
    Trigger concept: normalized random vector simulating 'Math'.
    """

    def __init__(self, latent_dim: int, seed: int = 123) -> None:
        self._latent_dim = latent_dim
        self._concept = _normalized_concept(latent_dim, seed)

    def name(self) -> str:
        return "Calculator"

    def trigger_concept(self) -> torch.Tensor:
        return self._concept

    def execute(self, *args: Any) -> Any:
        return "Calculating... [Mock Result]"
