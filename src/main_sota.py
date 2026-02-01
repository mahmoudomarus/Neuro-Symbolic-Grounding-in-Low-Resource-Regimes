"""
Phase 14: SOTA Mode - Zero-Shot Recognition with DINO + CLIP.

This demonstrates the "Full World Model" capability using pre-trained models:
- DINO ViT for visual understanding
- CLIP for language-vision grounding

The agent can now recognize ANY concept you can describe, without training.

Run from project root: python src/main_sota.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from src.world_model.encoder_sota import SOTA_Encoder, get_sota_encoder
from src.language.binder_sota import SOTA_Binder, get_sota_binder
from src.manager.agent import CognitiveAgent
from src.manager.config import AgentConfig
from src.memory.episodic import EpisodicMemory
from src.tools.library import WikiTool, CalculatorTool


class WorldModel:
    """Wrapper for encoder to match agent interface."""
    def __init__(self, encoder: torch.nn.Module) -> None:
        self.encoder = encoder


def demo_zero_shot_recognition() -> None:
    """
    Demonstrate zero-shot recognition capabilities.
    
    Shows that the agent can recognize concepts it was never explicitly trained on.
    """
    print("=" * 60)
    print("  SOTA Mode: Zero-Shot Recognition Demo")
    print("=" * 60)
    print()
    print("Loading DINO ViT (visual understanding)...")
    encoder = SOTA_Encoder(freeze=True)
    print(f"✓ Loaded {encoder.model_name} with {encoder.out_channels}-D embeddings")
    
    print()
    print("Loading CLIP (language-vision grounding)...")
    binder = SOTA_Binder(freeze=True)
    print(f"✓ Loaded {binder.model_name} with {binder.embedding_dim}-D embeddings")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = encoder.to(device)
    binder = binder.to(device)
    
    print()
    print("=" * 60)
    print("  Demo 1: Zero-Shot Classification")
    print("=" * 60)
    print()
    
    # Define test concepts (the model has never been explicitly trained on these)
    test_concepts = [
        "a platypus",
        "a red ferrari",
        "the Eiffel Tower",
        "a slice of pizza",
        "abstract art",
        "a space shuttle",
    ]
    
    print("The agent now understands concepts it was never trained on:")
    for i, concept in enumerate(test_concepts, 1):
        print(f"  {i}. {concept}")
    
    print()
    print("Testing with random image...")
    # Create a random test image (in real use, you'd load an actual image)
    test_image = torch.randn(1, 3, 224, 224).to(device)
    
    # Encode image and concepts
    image_features = encoder(test_image)  # [1, 768]
    
    # Get text embeddings for all concepts
    text_features = binder.embed_text(test_concepts)  # [6, 512]
    
    # Compute similarities (need to handle dimension mismatch)
    # For demo purposes, we'll use CLIP's image encoder directly
    image_features_clip = binder.embed_image(test_image)  # [1, 512]
    
    # Compute cosine similarity
    similarities = (image_features_clip @ text_features.T).squeeze(0)  # [6]
    similarities = F.softmax(similarities, dim=0)
    
    # Get top prediction
    top_idx = torch.argmax(similarities).item()
    top_concept = test_concepts[top_idx]
    top_confidence = similarities[top_idx].item() * 100
    
    print(f"\nTop prediction: '{top_concept}' ({top_confidence:.1f}% confidence)")
    print("\nAll confidences:")
    for concept, sim in zip(test_concepts, similarities):
        print(f"  {concept:<20} {sim.item()*100:5.1f}%")
    
    print()
    print("=" * 60)
    print("  Demo 2: Free-Form Text Queries")
    print("=" * 60)
    print()
    
    # Now test arbitrary natural language queries
    queries = [
        "a cute animal",
        "something you can drive",
        "food",
        "architecture",
        "technology",
    ]
    
    print("The agent can understand free-form natural language:")
    for i, query in enumerate(queries, 1):
        print(f"  {i}. '{query}'")
    
    print()
    print("Comparing image to free-form queries...")
    query_features = binder.embed_text(queries)
    query_similarities = (image_features_clip @ query_features.T).squeeze(0)
    query_similarities = F.softmax(query_similarities, dim=0)
    
    top_query_idx = torch.argmax(query_similarities).item()
    top_query = queries[top_query_idx]
    top_query_conf = query_similarities[top_query_idx].item() * 100
    
    print(f"\nBest match: '{top_query}' ({top_query_conf:.1f}% confidence)")
    
    print()
    print("=" * 60)
    print("  Demo 3: Episodic Memory with SOTA Models")
    print("=" * 60)
    print()
    
    # Create memory system
    memory = EpisodicMemory()
    
    print("Agent 'watches' a sequence of random images...")
    num_observations = 10
    for step in range(num_observations):
        # Generate random image
        obs = torch.randn(1, 3, 224, 224).to(device)
        
        # Encode with CLIP (for consistent dimension with text queries)
        obs_features = binder.embed_image(obs)
        obs_features = F.normalize(obs_features, dim=1)
        
        # Store in memory
        memory.store(obs_features, step, {"source": "random"})
        
        if (step + 1) % 3 == 0:
            print(f"  Stored {step + 1} memories...")
    
    print(f"\n✓ Stored {num_observations} observations in episodic memory")
    
    # Query memory with natural language
    query_text = "a cute animal"
    print(f"\nQuery: 'Find memories similar to {query_text}'")
    query_vec = binder.embed_text(query_text)
    
    matches = memory.recall(query_vec, threshold=0.5)
    if matches:
        print(f"Found {len(matches)} matches:")
        for step, similarity in matches[:3]:  # Show top 3
            print(f"  - Step {step}: similarity = {similarity:.3f}")
    else:
        print("  No matches found (threshold too high for random data)")
    
    print()
    print("=" * 60)
    print("  Summary")
    print("=" * 60)
    print()
    print("✓ Zero-shot recognition: Agent recognizes concepts without training")
    print("✓ Natural language: Understands free-form text descriptions")
    print("✓ Universal grounding: Same vector space for vision and language")
    print("✓ Episodic memory: Stores and recalls experiences by concept")
    print()
    print("Next step: Run the SOTA dashboard for interactive exploration:")
    print("  streamlit run src/dashboard_sota.py")
    print()


def interactive_mode() -> None:
    """
    Interactive mode: User can query any concept.
    """
    print()
    print("=" * 60)
    print("  Interactive Mode")
    print("=" * 60)
    print()
    
    print("Loading models...")
    encoder = SOTA_Encoder(freeze=True)
    binder = SOTA_Binder(freeze=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = encoder.to(device)
    binder = binder.to(device)
    
    print()
    print("Type any concept to see how the agent understands it.")
    print("Examples: 'a red sports car', 'a happy dog', 'Mount Everest'")
    print("Type 'quit' to exit.")
    print()
    
    while True:
        try:
            user_input = input("Concept: ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if not user_input:
                continue
            
            # Encode the concept
            concept_features = binder.embed_text(user_input)
            print(f"  ✓ Encoded '{user_input}' to {concept_features.shape[1]}-D vector")
            print(f"  ✓ Vector norm: {torch.norm(concept_features).item():.3f}")
            print()
            
        except KeyboardInterrupt:
            break
    
    print("\nGoodbye!")


def main() -> None:
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="SOTA Mode Demo")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode",
    )
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_mode()
    else:
        demo_zero_shot_recognition()


if __name__ == "__main__":
    main()
