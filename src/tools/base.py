"""
Abstract base class for tools (the "Dictionary" layer).

Every tool has a name, a trigger_concept vector (used to match latent state via
cosine similarity), and an execute method.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch


class Tool(ABC):
    """
    Abstract base for tools. Subclasses must implement name, trigger_concept,
    and execute. The trigger_concept vector defines the 'concept' that
    activates this tool (e.g. math, lookup).
    """

    @abstractmethod
    def name(self) -> str:
        """Return the tool's display name (e.g. for TOOL_CALL output)."""
        ...

    @abstractmethod
    def trigger_concept(self) -> torch.Tensor:
        """
        Return a vector representing the concept that triggers this tool.
        Shape must be (C,) or (1, C) so it can be compared to a pooled
        latent state of dimension C via cosine similarity.
        """
        ...

    @abstractmethod
    def execute(self, *args: Any) -> Any:
        """Run the tool with the given arguments. Return value is tool-specific."""
        ...
