"""
Dual Memory System.

Implements two memory systems like the human brain:
- Episodic Memory: Specific experiences (hippocampus)
- Semantic Memory: General knowledge (neocortex)

Key feature: Consolidation - repeated patterns in episodic memory
automatically transfer to semantic memory (like sleep consolidation).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import time

import torch
import torch.nn.functional as F


@dataclass
class MemoryEntry:
    """Single episodic memory entry."""
    vector: torch.Tensor
    timestamp: float
    metadata: Dict[str, Any]
    access_count: int = 0
    last_accessed: float = 0.0


@dataclass
class SemanticConcept:
    """Semantic memory concept (prototype)."""
    prototype: torch.Tensor
    count: int
    variance: float = 0.0
    last_updated: float = 0.0


class DualMemorySystem:
    """
    Two memory systems like the human brain:
    - Episodic: Specific experiences (hippocampus)
    - Semantic: General knowledge (neocortex)
    
    Consolidation: Repeated patterns in episodic -> semantic
    
    This enables:
    - Storing specific experiences
    - Abstracting to general concepts
    - Few-shot learning (new experiences quickly form concepts)
    """
    
    def __init__(
        self,
        dim: int = 512,
        max_episodic: int = 10000,
        consolidation_threshold: int = 5,
        similarity_threshold: float = 0.85,
        forget_rate: float = 0.01,
    ) -> None:
        """
        Initialize dual memory system.
        
        Args:
            dim: Feature dimension
            max_episodic: Maximum episodic memories to store
            consolidation_threshold: Number of similar episodes before consolidation
            similarity_threshold: Similarity threshold for matching
            forget_rate: Rate of forgetting old episodic memories
        """
        self.dim = dim
        self.max_episodic = max_episodic
        self.consolidation_threshold = consolidation_threshold
        self.similarity_threshold = similarity_threshold
        self.forget_rate = forget_rate
        
        # Episodic memory (specific experiences)
        self.episodic: List[MemoryEntry] = []
        
        # Semantic memory (general concepts)
        self.semantic: Dict[str, SemanticConcept] = {}
        
        # Pending consolidations (concept -> list of similar vectors)
        self._consolidation_buffer: Dict[str, List[torch.Tensor]] = {}
    
    def store(
        self,
        vector: torch.Tensor,
        metadata: Dict[str, Any],
        auto_consolidate: bool = True,
    ) -> None:
        """
        Store experience in episodic memory.
        
        Args:
            vector: Feature vector [D] or [1, D]
            metadata: Associated metadata (should include 'label' for consolidation)
            auto_consolidate: Whether to check for consolidation
        """
        # Normalize and detach
        vector = self._normalize(vector.detach().cpu())
        
        # Create memory entry
        now = time.time()
        entry = MemoryEntry(
            vector=vector,
            timestamp=now,
            metadata=metadata.copy(),
            last_accessed=now,
        )
        
        # Store in episodic
        self.episodic.append(entry)
        
        # Enforce max size (forget oldest, least accessed)
        if len(self.episodic) > self.max_episodic:
            self._forget_oldest()
        
        # Auto-consolidate if label provided
        if auto_consolidate and 'label' in metadata:
            self._try_consolidate(metadata['label'], vector)
    
    def _normalize(self, vector: torch.Tensor) -> torch.Tensor:
        """Normalize vector to unit length."""
        vector = vector.flatten()
        return F.normalize(vector.unsqueeze(0), dim=1).squeeze(0)
    
    def _forget_oldest(self) -> None:
        """Remove oldest, least accessed memories."""
        if not self.episodic:
            return
        
        # Score by recency and access count
        now = time.time()
        scores = []
        for entry in self.episodic:
            recency = 1.0 / (1.0 + now - entry.timestamp)
            access = entry.access_count / 10.0
            scores.append(recency + access)
        
        # Remove lowest scoring
        min_idx = scores.index(min(scores))
        self.episodic.pop(min_idx)
    
    def _try_consolidate(self, label: str, vector: torch.Tensor) -> None:
        """Try to consolidate to semantic memory if pattern repeats."""
        # Add to consolidation buffer
        if label not in self._consolidation_buffer:
            self._consolidation_buffer[label] = []
        self._consolidation_buffer[label].append(vector)
        
        # Check if we have enough similar examples
        if len(self._consolidation_buffer[label]) >= self.consolidation_threshold:
            # Compute prototype as mean of similar vectors
            vectors = torch.stack(self._consolidation_buffer[label])
            prototype = vectors.mean(dim=0)
            prototype = self._normalize(prototype)
            
            # Compute variance
            variance = vectors.var(dim=0).mean().item()
            
            # Store or update semantic concept
            if label in self.semantic:
                # Running average with existing prototype
                old = self.semantic[label]
                new_count = old.count + len(self._consolidation_buffer[label])
                weight = old.count / new_count
                
                new_prototype = weight * old.prototype + (1 - weight) * prototype
                new_prototype = self._normalize(new_prototype)
                
                self.semantic[label] = SemanticConcept(
                    prototype=new_prototype,
                    count=new_count,
                    variance=(old.variance + variance) / 2,
                    last_updated=time.time(),
                )
            else:
                self.semantic[label] = SemanticConcept(
                    prototype=prototype,
                    count=len(self._consolidation_buffer[label]),
                    variance=variance,
                    last_updated=time.time(),
                )
            
            # Clear buffer
            self._consolidation_buffer[label] = []
    
    def recall_episodic(
        self,
        query: torch.Tensor,
        k: int = 5,
        threshold: Optional[float] = None,
    ) -> List[Tuple[float, Dict[str, Any]]]:
        """
        Recall similar episodic memories.
        
        Args:
            query: Query vector [D] or [1, D]
            k: Maximum number of results
            threshold: Minimum similarity (default: self.similarity_threshold * 0.8)
            
        Returns:
            List of (similarity, metadata) tuples, sorted by similarity
        """
        if not self.episodic:
            return []
        
        query = self._normalize(query.detach().cpu())
        threshold = threshold or (self.similarity_threshold * 0.8)
        
        # Compute similarities
        results = []
        now = time.time()
        
        for entry in self.episodic:
            sim = F.cosine_similarity(
                query.unsqueeze(0), 
                entry.vector.unsqueeze(0)
            ).item()
            
            if sim >= threshold:
                # Update access info
                entry.access_count += 1
                entry.last_accessed = now
                
                results.append((sim, entry.metadata.copy()))
        
        # Sort by similarity (descending) and return top k
        results.sort(key=lambda x: x[0], reverse=True)
        return results[:k]
    
    def recall_semantic(
        self,
        query: torch.Tensor,
        k: int = 3,
    ) -> List[Tuple[str, float]]:
        """
        Recall semantic concepts by similarity.
        
        Args:
            query: Query vector [D] or [1, D]
            k: Maximum number of results
            
        Returns:
            List of (concept_name, similarity) tuples
        """
        if not self.semantic:
            return []
        
        query = self._normalize(query.detach().cpu())
        
        results = []
        for label, concept in self.semantic.items():
            sim = F.cosine_similarity(
                query.unsqueeze(0),
                concept.prototype.unsqueeze(0)
            ).item()
            results.append((label, sim))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]
    
    def recall(
        self,
        query: torch.Tensor,
        mode: str = 'both',
    ) -> Dict[str, Any]:
        """
        Recall from memory.
        
        Args:
            query: Query vector
            mode: 'episodic', 'semantic', or 'both'
            
        Returns:
            Dict with 'episodic' and/or 'semantic' results
        """
        results = {}
        
        if mode in ('episodic', 'both'):
            results['episodic'] = self.recall_episodic(query)
        
        if mode in ('semantic', 'both'):
            results['semantic'] = self.recall_semantic(query)
        
        return results
    
    def learn_new_concept(
        self,
        vectors: torch.Tensor,
        concept_name: str,
    ) -> None:
        """
        Directly learn a new semantic concept from examples.
        
        This bypasses the consolidation buffer for immediate learning
        (useful for few-shot learning scenarios).
        
        Args:
            vectors: Example vectors [K, D]
            concept_name: Name for the concept
        """
        # Normalize vectors
        vectors = vectors.detach().cpu()
        if vectors.dim() == 1:
            vectors = vectors.unsqueeze(0)
        
        normalized = F.normalize(vectors, dim=1)
        
        # Compute prototype
        prototype = normalized.mean(dim=0)
        prototype = self._normalize(prototype)
        
        # Compute variance
        variance = normalized.var(dim=0).mean().item()
        
        # Store
        self.semantic[concept_name] = SemanticConcept(
            prototype=prototype,
            count=vectors.shape[0],
            variance=variance,
            last_updated=time.time(),
        )
    
    def forget_semantic(self, concept_name: str) -> bool:
        """Remove a semantic concept."""
        if concept_name in self.semantic:
            del self.semantic[concept_name]
            return True
        return False
    
    def clear_episodic(self) -> None:
        """Clear all episodic memories."""
        self.episodic.clear()
        self._consolidation_buffer.clear()
    
    def clear_all(self) -> None:
        """Clear all memories."""
        self.episodic.clear()
        self.semantic.clear()
        self._consolidation_buffer.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            'episodic_count': len(self.episodic),
            'semantic_count': len(self.semantic),
            'semantic_concepts': list(self.semantic.keys()),
            'consolidation_pending': {
                k: len(v) for k, v in self._consolidation_buffer.items()
            },
        }
    
    def save(self, path: str) -> None:
        """Save memory to disk."""
        torch.save({
            'episodic': [
                {
                    'vector': e.vector,
                    'timestamp': e.timestamp,
                    'metadata': e.metadata,
                    'access_count': e.access_count,
                    'last_accessed': e.last_accessed,
                }
                for e in self.episodic
            ],
            'semantic': {
                k: {
                    'prototype': v.prototype,
                    'count': v.count,
                    'variance': v.variance,
                    'last_updated': v.last_updated,
                }
                for k, v in self.semantic.items()
            },
            'config': {
                'dim': self.dim,
                'max_episodic': self.max_episodic,
                'consolidation_threshold': self.consolidation_threshold,
                'similarity_threshold': self.similarity_threshold,
            },
        }, path)
    
    def load(self, path: str) -> None:
        """Load memory from disk."""
        data = torch.load(path)
        
        # Restore episodic
        self.episodic = [
            MemoryEntry(
                vector=e['vector'],
                timestamp=e['timestamp'],
                metadata=e['metadata'],
                access_count=e['access_count'],
                last_accessed=e['last_accessed'],
            )
            for e in data['episodic']
        ]
        
        # Restore semantic
        self.semantic = {
            k: SemanticConcept(
                prototype=v['prototype'],
                count=v['count'],
                variance=v['variance'],
                last_updated=v['last_updated'],
            )
            for k, v in data['semantic'].items()
        }
        
        # Restore config
        config = data.get('config', {})
        self.dim = config.get('dim', self.dim)
        self.max_episodic = config.get('max_episodic', self.max_episodic)
