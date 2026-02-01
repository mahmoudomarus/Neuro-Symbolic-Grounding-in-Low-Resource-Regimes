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
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import copy

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


class LearnedGrounding(nn.Module):
    """
    Grounding table populated through interaction, not manual entry.
    
    CRITICAL DESIGN DECISION: No hard-coded concept groundings.
    All groundings are learned through the babbling phase.
    
    Protocol:
    1. Initialize EMPTY
    2. "Babbling phase": Agent interacts with objects in simulator
    3. Each interaction updates grounding:
       - strike object -> measure audio frequency -> infer hardness
       - lift object -> measure force/acceleration -> infer weight
       - etc.
    
    Evaluation should report "zero-shot after babbling" not "zero-shot".
    
    References:
    - O'Regan & Noë (2001). Sensorimotor theory of perceptual experience.
    - Smith & Gasser (2005). Development of embodied cognition.
    """
    
    # Property names for grounding
    PROPERTY_NAMES = [
        "hardness", "weight", "size", "animacy", "rigidity",
        "transparency", "roughness", "temperature", "containment"
    ]
    
    # Action -> Property mappings (which actions inform which properties)
    ACTION_PROPERTY_MAP = {
        'strike': ['hardness'],           # Audio frequency -> hardness
        'lift': ['weight'],               # Force/acceleration -> weight
        'push': ['weight'],               # Resistance -> weight
        'squeeze': ['hardness', 'rigidity'],  # Deformation -> hardness/rigidity
        'look': ['size', 'transparency'], # Visual -> size/transparency
        'drop': ['hardness', 'weight'],   # Impact sound -> hardness/weight
    }
    
    def __init__(self, property_dim: int = 9, hidden_dim: int = 256) -> None:
        super().__init__()
        
        self.property_dim = property_dim
        self.hidden_dim = hidden_dim
        
        # EMPTY grounding table - populated through interaction
        self.grounding_table: Dict[str, torch.Tensor] = {}
        
        # Confidence scores for each grounding (how many interactions)
        self.grounding_confidence: Dict[str, float] = {}
        
        # Learnable word embeddings for generalization
        self.word_embedding = nn.Embedding(1000, hidden_dim)  # Vocabulary size
        self.word_to_idx: Dict[str, int] = {}
        self.next_word_idx = 0
        
        # Property predictor for unseen words (learns from grounded words)
        self.property_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, property_dim),
            nn.Sigmoid(),
        )
        
        # Property encoders from sensory feedback
        self.hardness_from_audio = nn.Sequential(
            nn.Linear(2, 32),  # audio_frequency, audio_intensity
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        
        self.weight_from_force = nn.Sequential(
            nn.Linear(2, 32),  # force_required, acceleration
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
    
    def learn_from_interaction(
        self,
        object_id: str,
        action: str,
        sensory_feedback: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Update grounding based on actual interaction.
        
        This is how the agent builds its concept dictionary through
        sensorimotor contingencies, not supervision.
        
        Args:
            object_id: Identifier for the object (e.g., "rock_001")
            action: Action performed (e.g., "strike", "lift")
            sensory_feedback: Dict of sensory values from interaction
            
        Returns:
            Dict of inferred properties from this interaction
        """
        inferred = {}
        
        # Initialize if first interaction with this object
        if object_id not in self.grounding_table:
            self.grounding_table[object_id] = torch.full((self.property_dim,), 0.5)
            self.grounding_confidence[object_id] = 0.0
        
        # Infer properties based on action type
        if action == 'strike':
            if 'audio_frequency' in sensory_feedback:
                # High frequency = hard
                hardness = sensory_feedback['audio_frequency']
                inferred['hardness'] = hardness
                self._update_property(object_id, 0, hardness)
                
        elif action in ('lift', 'push'):
            if 'force_required' in sensory_feedback:
                weight = sensory_feedback['force_required']
                inferred['weight'] = weight
                self._update_property(object_id, 1, weight)
            elif 'resistance' in sensory_feedback:
                weight = sensory_feedback['resistance']
                inferred['weight'] = weight
                self._update_property(object_id, 1, weight)
                
        elif action == 'squeeze':
            if 'deformation' in sensory_feedback:
                # High deformation = soft (low hardness)
                hardness = 1.0 - sensory_feedback['deformation']
                inferred['hardness'] = hardness
                self._update_property(object_id, 0, hardness)
            if 'resistance' in sensory_feedback:
                rigidity = sensory_feedback['resistance']
                inferred['rigidity'] = rigidity
                self._update_property(object_id, 4, rigidity)
                
        elif action == 'drop':
            if 'impact_sound' in sensory_feedback:
                # Loud impact = hard and/or heavy
                impact = sensory_feedback['impact_sound']
                inferred['hardness'] = impact
                self._update_property(object_id, 0, impact)
        
        # Update confidence
        self.grounding_confidence[object_id] += 0.1
        self.grounding_confidence[object_id] = min(1.0, self.grounding_confidence[object_id])
        
        return inferred
    
    def _update_property(
        self,
        object_id: str,
        property_idx: int,
        value: float,
        learning_rate: float = 0.3,
    ) -> None:
        """Update a specific property with exponential moving average."""
        current = self.grounding_table[object_id][property_idx].item()
        new_value = (1 - learning_rate) * current + learning_rate * value
        self.grounding_table[object_id][property_idx] = new_value
    
    def _get_word_idx(self, word: str) -> int:
        """Get or create word index."""
        word_lower = word.lower().strip()
        if word_lower not in self.word_to_idx:
            self.word_to_idx[word_lower] = self.next_word_idx
            self.next_word_idx += 1
        return self.word_to_idx[word_lower]
    
    def forward(self, word: str) -> torch.Tensor:
        """
        Ground a word in perceptual properties.
        
        If the word has been grounded through interaction, return that.
        Otherwise, use the learned predictor to estimate.
        
        Args:
            word: The word to ground
            
        Returns:
            Property vector [property_dim]
        """
        word_lower = word.lower().strip()
        
        # Check if we have grounded this concept through interaction
        if word_lower in self.grounding_table:
            return self.grounding_table[word_lower]
        
        # Check if any object_id contains this word (partial match)
        for obj_id, props in self.grounding_table.items():
            if word_lower in obj_id.lower():
                return props
        
        # Unknown word - use learned predictor
        word_idx = self._get_word_idx(word_lower)
        word_idx = min(word_idx, 999)  # Clamp to vocab size
        
        embedding = self.word_embedding(torch.tensor([word_idx]))
        predicted = self.property_predictor(embedding)
        
        return predicted.squeeze(0)
    
    def ground_phrase(self, phrase: str) -> torch.Tensor:
        """Ground a phrase (multiple words)."""
        words = phrase.lower().split()
        
        if not words:
            return torch.full((self.property_dim,), 0.5)
        
        # Average properties of all words
        word_props = [self.forward(word) for word in words]
        
        if word_props:
            return torch.stack(word_props).mean(dim=0)
        
        return torch.full((self.property_dim,), 0.5)
    
    def get_grounded_concepts(self) -> List[str]:
        """Return list of concepts that have been grounded through interaction."""
        return list(self.grounding_table.keys())
    
    def get_grounding_statistics(self) -> Dict[str, Any]:
        """Get statistics about current grounding state."""
        return {
            'num_grounded': len(self.grounding_table),
            'concepts': list(self.grounding_table.keys()),
            'confidences': dict(self.grounding_confidence),
            'avg_confidence': (
                sum(self.grounding_confidence.values()) / len(self.grounding_confidence)
                if self.grounding_confidence else 0.0
            ),
        }
    
    def export_grounding_table(self) -> Dict[str, List[float]]:
        """Export grounding table for inspection/verification."""
        return {
            obj_id: props.tolist()
            for obj_id, props in self.grounding_table.items()
        }


# Backward compatibility alias
TextGrounder = LearnedGrounding


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
        
        # Language → Concept (learned through interaction, not hard-coded)
        self.grounder = LearnedGrounding(config.property_dim, config.hidden_dim)
        
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
        
        Searches through concepts grounded via babbling interaction.
        
        Returns:
            (best_word, similarity_score)
        """
        best_word = None
        best_score = 0.0
        
        # Get concepts that have been grounded through interaction
        grounded_concepts = self.grounder.get_grounded_concepts()
        
        if not grounded_concepts:
            # No concepts grounded yet - return unknown
            return "unknown", 0.0
        
        for word in grounded_concepts:
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
