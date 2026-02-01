"""
Episodic memory: store latent vectors and recall by similarity to a query.

Operates entirely in latent space; no raw images are stored.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


class EpisodicMemory:
    """
    Stores a sequence of experiences as normalized latent vectors.
    Recall returns steps where stored vectors match a query (cosine similarity).
    """

    def __init__(self) -> None:
        self._storage: list[dict[str, Any]] = []

    def store(
        self,
        observation_vector: torch.Tensor,
        step: int,
        metadata: Any = None,
    ) -> None:
        """
        Store a pooled latent vector for this step.
        Detaches and moves to CPU to save memory; normalizes before storing.
        """
        v = observation_vector.detach().cpu()
        if v.dim() > 1:
            v = v.squeeze(0)
        v = F.normalize(v.unsqueeze(0), dim=1).squeeze(0)
        self._storage.append({
            "vector": v,
            "timestamp": step,
            "meta": metadata,
        })

    def recall(
        self,
        query_vector: torch.Tensor,
        threshold: float = 0.8,
    ) -> list[tuple[int, float]]:
        """
        Compute cosine similarity of query against all stored vectors.
        Returns list of (step, similarity) for matches >= threshold.
        """
        q = query_vector.detach().cpu()
        if q.dim() > 1:
            q = q.squeeze(0)
        q = F.normalize(q.unsqueeze(0), dim=1)

        matches: list[tuple[int, float]] = []
        for entry in self._storage:
            v = entry["vector"].unsqueeze(0)
            sim = F.cosine_similarity(q, v, dim=1).item()
            if sim >= threshold:
                matches.append((entry["timestamp"], sim))
        matches.sort(key=lambda x: x[0])
        return matches

    def clear(self) -> None:
        """Reset memory (empty storage)."""
        self._storage.clear()

    def get_recent(self, n: int = 10) -> list[dict[str, Any]]:
        """Return the last n entries (for display). Each entry has keys: timestamp, meta, vector."""
        return list(self._storage[-n:])
