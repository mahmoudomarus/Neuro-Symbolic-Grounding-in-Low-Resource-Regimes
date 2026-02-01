"""
Phase 3 tests: Cognitive agent and tool interface.

Mock world model and tools. Test 1: strong (high-confidence) latent -> ACTION.
Test 2: weak/noisy latent -> TOOL_CALL.
"""
from __future__ import annotations

from typing import Any

import torch

from src.manager.agent import CognitiveAgent
from src.manager.config import AgentConfig
from src.tools.base import Tool


class MockWorldModel:
    """World model that returns a configurable latent tensor (for testing)."""

    def __init__(self, latent_output: torch.Tensor) -> None:
        self.latent_output = latent_output

    def encoder(self, observation: torch.Tensor) -> torch.Tensor:
        return self.latent_output


class MockTool(Tool):
    """Concrete tool for tests; trigger_concept matches latent dim C."""

    def __init__(self, tool_name: str, concept: torch.Tensor) -> None:
        self._name = tool_name
        self._concept = concept

    def name(self) -> str:
        return self._name

    def trigger_concept(self) -> torch.Tensor:
        return self._concept

    def execute(self, *args: Any) -> Any:
        return "mock_result"


def test_agent_returns_action_when_confident() -> None:
    """Strong latent (high magnitude, low variance) -> agent returns ACTION."""
    # High magnitude, low variance -> low uncertainty -> u < 0.7
    C, H, W = 64, 4, 4
    strong_latent = torch.ones(1, C, H, W) * 5.0  # magnitude ~ 5 * sqrt(C*H*W)
    world_model = MockWorldModel(strong_latent)
    concept = torch.randn(C)
    tool = MockTool("Calculator", concept)
    config = AgentConfig(uncertainty_threshold=0.7)
    agent = CognitiveAgent(world_model, [tool], config)

    dummy_obs = torch.zeros(1, 3, 32, 32)
    result, _ = agent.step(dummy_obs)

    assert result.startswith("ACTION:")
    assert "MOVE" in result


def test_agent_returns_tool_call_when_uncertain() -> None:
    """Weak/noisy latent (low magnitude) -> high u -> agent returns TOOL_CALL."""
    C, H, W = 64, 4, 4
    # Faint: near-zero magnitude -> high u_mag -> u above threshold
    weak_latent = torch.zeros(1, C, H, W) + 0.01
    world_model = MockWorldModel(weak_latent)
    concept = torch.randn(C)
    tool = MockTool("Wikipedia", concept)
    config = AgentConfig(uncertainty_threshold=0.7)
    agent = CognitiveAgent(world_model, [tool], config)

    dummy_obs = torch.zeros(1, 3, 32, 32)
    result, _ = agent.step(dummy_obs)

    assert result.startswith("TOOL_CALL:")
    assert "Wikipedia" in result


def test_agent_selects_tool_by_cosine_similarity() -> None:
    """When uncertain, agent picks the tool whose trigger_concept is most similar to latent."""
    C, H, W = 8, 2, 2
    # Latent pooled to [C]; make it align with concept_a
    target_vec = torch.randn(C)
    weak_latent = target_vec.reshape(1, C, 1, 1).expand(1, C, H, W)
    world_model = MockWorldModel(weak_latent)
    concept_a = target_vec.clone()
    concept_b = torch.randn(C)  # different
    tool_a = MockTool("ToolA", concept_a)
    tool_b = MockTool("ToolB", concept_b)
    config = AgentConfig(uncertainty_threshold=0.1)  # force tool branch
    agent = CognitiveAgent(world_model, [tool_a, tool_b], config)

    dummy_obs = torch.zeros(1, 3, 16, 16)
    result, _ = agent.step(dummy_obs)

    assert "TOOL_CALL:" in result
    assert "ToolA" in result
