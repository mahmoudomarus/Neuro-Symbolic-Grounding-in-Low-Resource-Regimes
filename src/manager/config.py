"""
Configuration for the manager (agent) layer.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class AgentConfig:
    """Configuration for the cognitive agent (System 2 loop)."""

    uncertainty_threshold: float = 0.7  # Below: act directly. At or above: use tools.
