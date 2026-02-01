"""
Cognitive agent: System 2 loop that decides to act or use tools based on uncertainty.

Encodes observation to latent state, measures uncertainty (heuristic: magnitude
and variance), then either returns a direct action (confident) or selects a tool
via cosine similarity to trigger concepts (confused).
"""
from __future__ import annotations

from typing import Any, List, Protocol, Union

import torch
import torch.nn.functional as F

from src.tools.base import Tool

from .config import AgentConfig


class WorldModelProtocol(Protocol):
    """Protocol for world model: must expose an encoder mapping observation -> latent."""

    def encoder(self, observation: torch.Tensor) -> torch.Tensor:
        ...


class CognitiveAgent:
    """
    Agent that either acts directly (System 1) or calls a tool (System 2) based
    on latent uncertainty. Uses a heuristic for uncertainty; can be replaced
    later by a learned head.
    """

    def __init__(
        self,
        world_model: WorldModelProtocol,
        tools: List[Tool],
        config: AgentConfig,
    ) -> None:
        self.world_model = world_model
        self.tools = list(tools)
        self.config = config

    def measure_uncertainty(self, latent_state: torch.Tensor) -> float:
        """
        Heuristic uncertainty from latent state [B, C, H, W].
        High when vector is 'faint' (low L2 magnitude) or 'noisy' (high variance).
        Returns a scalar in roughly [0, 1]; high means uncertain.
        """
        # Per-sample L2 norm of flattened latent, then mean over batch
        magnitude = latent_state.flatten(1).norm(dim=1).mean().item()
        variance = latent_state.var().item()
        # Low magnitude ('faint') or high variance ('noisy') -> high uncertainty
        u_mag = 1.0 / (1.0 + magnitude)
        u_var = min(1.0, variance / 10.0)
        # Max: either faint or noisy is enough to be uncertain
        u = max(u_mag, u_var)
        return float(u)

    def _pool_latent(self, z: torch.Tensor) -> torch.Tensor:
        """Pool [B, C, H, W] to [C] for comparison with trigger_concept."""
        return z.mean(dim=(0, 2, 3))

    def _best_tool_for_latent(self, z_vec: torch.Tensor) -> Tool:
        """Select tool with highest cosine similarity between z_vec and trigger_concept."""
        if not self.tools:
            raise RuntimeError("No tools registered; cannot select tool.")
        best_tool: Union[Tool, None] = None
        best_sim = -2.0
        for tool in self.tools:
            concept = tool.trigger_concept()
            if concept.dim() > 1:
                concept = concept.flatten()
            # Ensure same device and expand to same dims for cosine_similarity
            concept = concept.to(z_vec.device).to(z_vec.dtype)
            if concept.shape[0] != z_vec.shape[0]:
                continue
            sim = F.cosine_similarity(
                z_vec.unsqueeze(0),
                concept.unsqueeze(0),
                dim=1,
            ).item()
            if sim > best_sim:
                best_sim = sim
                best_tool = tool
        if best_tool is None:
            raise RuntimeError("No tool matched latent dimension; cannot select tool.")
        return best_tool

    def step(self, observation: torch.Tensor) -> tuple[str, float]:
        """
        One cognitive step: encode observation, check uncertainty, then act or call tool.

        Args:
            observation: Observation tensor [B, C_in, H, W] (e.g. pixels).

        Returns:
            (decision_string, uncertainty): e.g. ("ACTION: MOVE", 0.2) or
            ("TOOL_CALL: Wikipedia", 0.8).
        """
        with torch.no_grad():
            z_t = self.world_model.encoder(observation)
        u = self.measure_uncertainty(z_t)
        threshold = self.config.uncertainty_threshold

        if u < threshold:
            print("  [Branch: System 1 (Instinct)]")
            return "ACTION: MOVE", u
        # System 2: compare to tool trigger concepts
        print("  [Branch: System 2 (Reasoning)]")
        z_vec = self._pool_latent(z_t)
        tool = self._best_tool_for_latent(z_vec)
        return f"TOOL_CALL: {tool.name()}", u
