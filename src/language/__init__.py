"""
Language module: concept binding and LLM integration.

Includes:
- ConceptBinder: Original class label grounding in encoder space
- LanguageGrounding: Bi-directional grounding with LLM integration
"""
from .binder import ConceptBinder
from .llm_integration import (
    LanguageGrounding,
    LanguageConfig,
    ConceptVerbalizer,
    TextGrounder,
    LLMInterface,
)

__all__ = [
    "ConceptBinder",
    "LanguageGrounding",
    "LanguageConfig",
    "ConceptVerbalizer",
    "TextGrounder",
    "LLMInterface",
]
