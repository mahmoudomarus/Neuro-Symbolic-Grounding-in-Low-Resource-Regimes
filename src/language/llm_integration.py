"""
LLM Integration - Connect perceptual concepts to language.

This module bridges our grounded world model with large language models:
- Concept → Language: Describe what we perceive in words
- Language → Concept: Ground words in perceptual experience
- Reasoning: Use LLM for complex linguistic reasoning

Key insight: Language is NOT the foundation - it's a tool built on top
of grounded perceptual concepts. The LLM serves as a "semantic dictionary"
that we query, not as the source of understanding.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import json

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LanguageConfig:
    """Configuration for language integration."""
    concept_dim: int = 256
    property_dim: int = 9
    hidden_dim: int = 512
    vocab_embedding_dim: int = 768  # Common LLM embedding size
    use_external_llm: bool = False
    llm_model: str = "gpt-3.5-turbo"


class ConceptVerbalizer(nn.Module):
    """
    Convert perceptual concepts to language descriptions.
    
    Takes a property vector (hard, heavy, small, etc.) and
    generates a language description.
    
    This is NOT generative text - it's structured description
    based on property values.
    """
    
    # Property names for verbalization
    PROPERTY_NAMES = [
        "hardness", "weight", "size", "animacy", "rigidity",
        "transparency", "roughness", "temperature", "containment"
    ]
    
    # Adjectives for each property at low/high values
    PROPERTY_ADJECTIVES = {
        "hardness": ("soft", "hard"),
        "weight": ("light", "heavy"),
        "size": ("small", "large"),
        "animacy": ("inanimate", "animate"),
        "rigidity": ("flexible", "rigid"),
        "transparency": ("opaque", "transparent"),
        "roughness": ("smooth", "rough"),
        "temperature": ("cold", "hot"),
        "containment": ("solid", "hollow"),
    }
    
    def __init__(self, property_dim: int = 9) -> None:
        super().__init__()
        self.property_dim = property_dim
    
    def forward(
        self,
        property_vector: torch.Tensor,
        threshold: float = 0.3,
    ) -> List[str]:
        """
        Generate language description from property vector.
        
        Args:
            property_vector: [B, property_dim] or [property_dim]
            threshold: Only describe properties with strong values
            
        Returns:
            List of descriptions (one per batch item)
        """
        if property_vector.dim() == 1:
            property_vector = property_vector.unsqueeze(0)
        
        B = property_vector.shape[0]
        descriptions = []
        
        for b in range(B):
            props = property_vector[b]
            desc_parts = []
            
            for i, (name, (low_adj, high_adj)) in enumerate(
                zip(self.PROPERTY_NAMES, self.PROPERTY_ADJECTIVES.values())
            ):
                if i >= len(props):
                    break
                    
                val = props[i].item()
                
                if val < threshold:
                    desc_parts.append(low_adj)
                elif val > (1 - threshold):
                    desc_parts.append(high_adj)
                # Middle values not described
            
            if desc_parts:
                description = "A " + ", ".join(desc_parts) + " object"
            else:
                description = "An object with moderate properties"
            
            descriptions.append(description)
        
        return descriptions
    
    def property_to_description(
        self,
        property_name: str,
        value: float,
    ) -> str:
        """Describe a single property."""
        if property_name not in self.PROPERTY_ADJECTIVES:
            return f"{property_name}={value:.2f}"
        
        low_adj, high_adj = self.PROPERTY_ADJECTIVES[property_name]
        
        if value < 0.3:
            return low_adj
        elif value > 0.7:
            return high_adj
        else:
            return f"moderately {low_adj}/{high_adj}"


class TextGrounder(nn.Module):
    """
    Ground language descriptions in perceptual properties.
    
    Takes text like "rock" and produces expected property values:
    - "rock" → hard=0.9, heavy=0.7, small=0.3, animate=0.0
    
    Uses learned word → property mappings.
    """
    
    # Pre-defined groundings for common concepts
    CONCEPT_GROUNDINGS = {
        "rock": [0.9, 0.7, 0.3, 0.0, 0.9, 0.0, 0.7, 0.5, 0.0],
        "water": [0.0, 0.3, 0.5, 0.0, 0.0, 0.8, 0.0, 0.4, 0.0],
        "animal": [0.5, 0.5, 0.5, 1.0, 0.3, 0.0, 0.5, 0.6, 0.0],
        "ball": [0.5, 0.3, 0.3, 0.0, 0.8, 0.0, 0.2, 0.5, 0.0],
        "glass": [0.9, 0.3, 0.4, 0.0, 0.9, 0.9, 0.0, 0.5, 1.0],
        "cloth": [0.1, 0.1, 0.5, 0.0, 0.0, 0.0, 0.7, 0.5, 0.0],
        "metal": [1.0, 0.9, 0.5, 0.0, 1.0, 0.0, 0.3, 0.5, 0.0],
        "wood": [0.8, 0.6, 0.5, 0.0, 0.8, 0.0, 0.6, 0.5, 0.0],
        "person": [0.6, 0.6, 0.7, 1.0, 0.3, 0.0, 0.5, 0.6, 0.0],
        "cat": [0.4, 0.3, 0.3, 1.0, 0.2, 0.0, 0.4, 0.6, 0.0],
        "dog": [0.5, 0.5, 0.5, 1.0, 0.2, 0.0, 0.5, 0.6, 0.0],
        "chair": [0.7, 0.6, 0.6, 0.0, 0.8, 0.0, 0.5, 0.5, 0.0],
        "table": [0.8, 0.7, 0.7, 0.0, 0.9, 0.0, 0.4, 0.5, 0.0],
        "cup": [0.6, 0.2, 0.2, 0.0, 0.8, 0.3, 0.3, 0.5, 1.0],
    }
    
    def __init__(self, property_dim: int = 9, hidden_dim: int = 256) -> None:
        super().__init__()
        
        self.property_dim = property_dim
        
        # Convert pre-defined groundings to buffer
        concept_names = list(self.CONCEPT_GROUNDINGS.keys())
        concept_vectors = torch.tensor([
            self.CONCEPT_GROUNDINGS[name] for name in concept_names
        ])
        self.register_buffer('known_concepts', concept_vectors)
        self.concept_names = concept_names
        
        # For unknown words, predict from word embedding (simplified)
        self.unknown_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, property_dim),
            nn.Sigmoid(),
        )
    
    def forward(self, word: str) -> torch.Tensor:
        """
        Ground a word in perceptual properties.
        
        Args:
            word: The word to ground
            
        Returns:
            Property vector [property_dim]
        """
        word_lower = word.lower().strip()
        
        # Check known concepts
        if word_lower in self.concept_names:
            idx = self.concept_names.index(word_lower)
            return self.known_concepts[idx]
        
        # Unknown word - return neutral properties
        return torch.full((self.property_dim,), 0.5)
    
    def ground_phrase(self, phrase: str) -> torch.Tensor:
        """Ground a phrase (multiple words)."""
        words = phrase.lower().split()
        
        if not words:
            return torch.full((self.property_dim,), 0.5)
        
        # Average properties of known words
        known_props = []
        for word in words:
            if word in self.concept_names:
                idx = self.concept_names.index(word)
                known_props.append(self.known_concepts[idx])
        
        if known_props:
            return torch.stack(known_props).mean(dim=0)
        
        return torch.full((self.property_dim,), 0.5)


class LLMInterface:
    """
    Interface to external LLM for complex reasoning.
    
    The LLM serves as a "semantic dictionary" - we query it for
    information, but understanding comes from our grounded model.
    """
    
    def __init__(self, model: str = "gpt-3.5-turbo", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key
        self._client = None
    
    def _get_client(self):
        """Lazily initialize OpenAI client."""
        if self._client is None:
            try:
                import openai
                self._client = openai.OpenAI(api_key=self.api_key)
            except ImportError:
                return None
        return self._client
    
    def query(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful assistant that answers questions about objects and their properties concisely.",
    ) -> Optional[str]:
        """Query the LLM."""
        client = self._get_client()
        if client is None:
            return None
        
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=100,
                temperature=0.3,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"LLM query failed: {e}")
            return None
    
    def get_property_info(self, concept: str, property_name: str) -> Optional[str]:
        """Ask LLM about a specific property of a concept."""
        prompt = f"What is the typical {property_name} of a {concept}? Answer in one short sentence."
        return self.query(prompt)
    
    def compare_concepts(self, concept1: str, concept2: str) -> Optional[str]:
        """Ask LLM to compare two concepts."""
        prompt = f"How are {concept1} and {concept2} similar and different? Answer briefly."
        return self.query(prompt)


class LanguageGrounding(nn.Module):
    """
    Bi-directional grounding between concepts and language.
    
    This is the bridge between our perceptual world model and language:
    - Concept → Language: Describe what we perceive
    - Language → Concept: Understand what words mean perceptually
    - Use LLM for complex reasoning, grounded in perception
    """
    
    def __init__(self, config: LanguageConfig) -> None:
        super().__init__()
        
        self.config = config
        
        # Concept → Language
        self.verbalizer = ConceptVerbalizer(config.property_dim)
        
        # Language → Concept
        self.grounder = TextGrounder(config.property_dim, config.hidden_dim)
        
        # LLM interface (optional)
        if config.use_external_llm:
            self.llm = LLMInterface(config.llm_model)
        else:
            self.llm = None
        
        # Concept similarity for matching
        self.concept_matcher = nn.Sequential(
            nn.Linear(config.property_dim * 2, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, 1),
            nn.Sigmoid(),
        )
    
    def describe_concept(
        self,
        property_vector: torch.Tensor,
    ) -> List[str]:
        """Convert perceptual concept to language description."""
        return self.verbalizer(property_vector)
    
    def ground_word(self, word: str) -> torch.Tensor:
        """Get expected perceptual properties for a word."""
        return self.grounder(word)
    
    def concept_matches_word(
        self,
        property_vector: torch.Tensor,
        word: str,
        threshold: float = 0.7,
    ) -> Tuple[bool, float]:
        """
        Check if a perceived concept matches a word.
        
        Args:
            property_vector: Perceived properties
            word: Word to check against
            threshold: Match threshold
            
        Returns:
            (matches, similarity_score)
        """
        expected = self.grounder(word)
        
        if property_vector.dim() == 1:
            property_vector = property_vector.unsqueeze(0)
        
        expected = expected.unsqueeze(0).expand(property_vector.shape[0], -1)
        
        # Compute similarity
        combined = torch.cat([property_vector, expected], dim=-1)
        similarity = self.concept_matcher(combined).squeeze(-1)
        
        matches = similarity > threshold
        
        return matches[0].item(), similarity[0].item()
    
    def find_matching_word(
        self,
        property_vector: torch.Tensor,
    ) -> Tuple[str, float]:
        """
        Find the best matching word for a property vector.
        
        Returns:
            (best_word, similarity_score)
        """
        best_word = None
        best_score = 0.0
        
        for word in self.grounder.concept_names:
            matches, score = self.concept_matches_word(property_vector, word)
            if score > best_score:
                best_score = score
                best_word = word
        
        return best_word or "unknown", best_score
    
    def reason_with_llm(
        self,
        perceived_description: str,
        question: str,
    ) -> str:
        """
        Use LLM for reasoning, grounded in perception.
        
        Our model provides the perceptual description,
        LLM does linguistic reasoning.
        """
        if self.llm is None:
            return "LLM not available"
        
        prompt = f"""
Based on this perception: "{perceived_description}"

Question: {question}

Provide a brief, grounded answer based on typical properties of such objects.
"""
        
        response = self.llm.query(prompt)
        return response or "Could not get LLM response"
    
    def answer_property_question(
        self,
        property_vector: torch.Tensor,
        question: str,
    ) -> str:
        """
        Answer a question about perceived object.
        
        Combines perceptual grounding with LLM reasoning.
        """
        # First, describe what we perceive
        descriptions = self.describe_concept(property_vector)
        description = descriptions[0] if descriptions else "An object"
        
        # Find matching concept
        matched_word, confidence = self.find_matching_word(property_vector)
        
        if confidence > 0.7:
            description = f"{description} (likely a {matched_word})"
        
        # Use LLM if available
        if self.llm is not None:
            return self.reason_with_llm(description, question)
        
        # Fallback: Simple rule-based answer
        return f"Based on perception: {description}. Cannot answer '{question}' without LLM."
