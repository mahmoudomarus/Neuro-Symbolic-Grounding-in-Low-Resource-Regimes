#!/usr/bin/env python3
"""
Test script for SOTA models (DINO + CLIP).

Verifies that zero-shot recognition works correctly.

Run from project root: python test_sota.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import torch
import torch.nn.functional as F


def test_imports() -> bool:
    """Test that all SOTA modules can be imported."""
    print("=" * 60)
    print("Test 1: Imports")
    print("=" * 60)
    
    try:
        from src.world_model.encoder_sota import SOTA_Encoder, get_sota_encoder
        print("✓ SOTA_Encoder imported successfully")
        
        from src.language.binder_sota import SOTA_Binder, get_sota_binder
        print("✓ SOTA_Binder imported successfully")
        
        import timm
        print(f"✓ timm version: {timm.__version__}")
        
        import open_clip
        print(f"✓ open_clip imported successfully")
        
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        print("\nInstall required packages:")
        print("  pip install timm open-clip-torch")
        return False


def test_encoder() -> bool:
    """Test DINO encoder."""
    print("\n" + "=" * 60)
    print("Test 2: DINO Encoder")
    print("=" * 60)
    
    try:
        from src.world_model.encoder_sota import SOTA_Encoder
        
        print("Creating SOTA_Encoder...")
        encoder = SOTA_Encoder(freeze=True)
        print(f"✓ Model: {encoder.model_name}")
        print(f"✓ Embedding dim: {encoder.out_channels}")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder = encoder.to(device)
        print(f"✓ Device: {device}")
        
        # Test forward pass
        print("\nTesting forward pass...")
        test_input = torch.randn(2, 3, 224, 224).to(device)
        with torch.no_grad():
            output = encoder(test_input)
        
        print(f"✓ Input shape: {test_input.shape}")
        print(f"✓ Output shape: {output.shape}")
        
        assert output.shape == (2, 768), f"Expected (2, 768), got {output.shape}"
        print("✓ Output shape is correct")
        
        # Test with different input size
        print("\nTesting with different input size...")
        test_input_small = torch.randn(1, 3, 64, 64).to(device)
        with torch.no_grad():
            output_small = encoder(test_input_small)
        
        print(f"✓ Input shape: {test_input_small.shape}")
        print(f"✓ Output shape: {output_small.shape}")
        assert output_small.shape == (1, 768), f"Expected (1, 768), got {output_small.shape}"
        print("✓ Auto-resize works correctly")
        
        return True
    except Exception as e:
        print(f"✗ Encoder test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_binder() -> bool:
    """Test CLIP binder."""
    print("\n" + "=" * 60)
    print("Test 3: CLIP Binder")
    print("=" * 60)
    
    try:
        from src.language.binder_sota import SOTA_Binder
        
        print("Creating SOTA_Binder...")
        binder = SOTA_Binder(freeze=True)
        print(f"✓ Model: {binder.model_name}")
        print(f"✓ Embedding dim: {binder.embedding_dim}")
        print(f"✓ Device: {binder.device}")
        
        # Test text embedding
        print("\nTesting text embedding...")
        test_texts = ["a red ferrari", "a cute cat", "a tall building"]
        text_features = binder.embed_text(test_texts)
        
        print(f"✓ Input: {len(test_texts)} texts")
        print(f"✓ Output shape: {text_features.shape}")
        assert text_features.shape[0] == 3, f"Expected 3 texts, got {text_features.shape[0]}"
        print("✓ Text embedding works")
        
        # Verify normalization
        norms = torch.norm(text_features, dim=1)
        print(f"✓ Vector norms: {norms.tolist()}")
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), "Vectors not normalized"
        print("✓ Vectors are normalized")
        
        # Test image embedding
        print("\nTesting image embedding...")
        device = torch.device(binder.device)
        test_images = torch.randn(2, 3, 224, 224).to(device)
        image_features = binder.embed_image(test_images)
        
        print(f"✓ Input shape: {test_images.shape}")
        print(f"✓ Output shape: {image_features.shape}")
        assert image_features.shape[0] == 2, f"Expected 2 images, got {image_features.shape[0]}"
        print("✓ Image embedding works")
        
        # Test classification
        print("\nTesting zero-shot classification...")
        class_names = ["dog", "cat", "car", "building"]
        pred_indices, probs = binder.classify(test_images, class_names)
        
        print(f"✓ Classes: {class_names}")
        print(f"✓ Predictions: {[class_names[i] for i in pred_indices.tolist()]}")
        print(f"✓ Probabilities shape: {probs.shape}")
        
        # Verify probabilities sum to 1
        prob_sums = probs.sum(dim=1)
        print(f"✓ Probability sums: {prob_sums.tolist()}")
        assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5), "Probabilities don't sum to 1"
        print("✓ Classification works correctly")
        
        return True
    except Exception as e:
        print(f"✗ Binder test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_zero_shot() -> bool:
    """Test zero-shot recognition on novel concepts."""
    print("\n" + "=" * 60)
    print("Test 4: Zero-Shot Recognition")
    print("=" * 60)
    
    try:
        from src.world_model.encoder_sota import SOTA_Encoder
        from src.language.binder_sota import SOTA_Binder
        
        print("Loading models...")
        encoder = SOTA_Encoder(freeze=True)
        binder = SOTA_Binder(freeze=True)
        
        device = torch.device(binder.device)
        encoder = encoder.to(device)
        
        # Test with novel concepts that weren't in the original training
        novel_concepts = [
            "a platypus",
            "the Eiffel Tower",
            "abstract expressionism",
            "a quantum computer",
            "medieval armor",
        ]
        
        print(f"\n✓ Testing with {len(novel_concepts)} novel concepts:")
        for i, concept in enumerate(novel_concepts, 1):
            print(f"  {i}. {concept}")
        
        # Create random test image
        test_image = torch.randn(1, 3, 224, 224).to(device)
        
        # Classify using novel concepts
        pred_indices, probs = binder.classify(test_image, novel_concepts)
        
        top_concept = novel_concepts[pred_indices[0].item()]
        top_prob = probs[0, pred_indices[0]].item() * 100
        
        print(f"\n✓ Top prediction: '{top_concept}' ({top_prob:.1f}%)")
        print("\n✓ All predictions:")
        for concept, prob in zip(novel_concepts, probs[0]):
            print(f"  {concept:<25} {prob.item()*100:5.1f}%")
        
        print("\n✓ Zero-shot recognition works on novel concepts!")
        
        return True
    except Exception as e:
        print(f"✗ Zero-shot test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cross_modal() -> bool:
    """Test cross-modal similarity (image-text)."""
    print("\n" + "=" * 60)
    print("Test 5: Cross-Modal Similarity")
    print("=" * 60)
    
    try:
        from src.language.binder_sota import SOTA_Binder
        
        binder = SOTA_Binder(freeze=True)
        device = torch.device(binder.device)
        
        # Create test image
        test_image = torch.randn(1, 3, 224, 224).to(device)
        
        # Encode image
        image_features = binder.embed_image(test_image)
        
        # Test various text descriptions
        descriptions = [
            "a photograph",
            "random noise",
            "a natural scene",
            "geometric patterns",
        ]
        
        print(f"\n✓ Comparing image to {len(descriptions)} descriptions:")
        
        for desc in descriptions:
            text_features = binder.embed_text(desc)
            similarity = (image_features @ text_features.T).item()
            match_pct = (similarity + 1) / 2 * 100
            
            print(f"  '{desc}':")
            print(f"    Similarity: {similarity:.3f} ({match_pct:.1f}%)")
        
        print("\n✓ Cross-modal similarity computation works!")
        
        return True
    except Exception as e:
        print(f"✗ Cross-modal test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main() -> None:
    """Run all tests."""
    print("\n" + "=" * 60)
    print("  SOTA Models Test Suite")
    print("=" * 60)
    print()
    
    results = []
    
    # Run tests
    results.append(("Imports", test_imports()))
    
    if results[0][1]:  # Only continue if imports succeed
        results.append(("DINO Encoder", test_encoder()))
        results.append(("CLIP Binder", test_binder()))
        results.append(("Zero-Shot Recognition", test_zero_shot()))
        results.append(("Cross-Modal Similarity", test_cross_modal()))
    
    # Summary
    print("\n" + "=" * 60)
    print("  Test Summary")
    print("=" * 60)
    print()
    
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name:<30} {status}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    
    print()
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests passed! SOTA models are working correctly.")
        print("\nNext steps:")
        print("  1. Run the demo: python src/main_sota.py")
        print("  2. Launch dashboard: streamlit run src/dashboard_sota.py")
    else:
        print("\n✗ Some tests failed. Check the output above for details.")
        print("\nTroubleshooting:")
        print("  - Ensure all dependencies are installed: pip install -r requirements.txt")
        print("  - Check your internet connection (models download on first run)")
        print("  - Try running with --verbose flag for more details")
        sys.exit(1)


if __name__ == "__main__":
    main()
