# Phase 14: SOTA Upgrade — Zero-Shot Recognition

## What Changed?

We've upgraded from our custom "kindergarten brain" to **state-of-the-art pre-trained models**:

| Component | Before (Phase 1-12) | After (Phase 14) |
|-----------|---------------------|------------------|
| **Visual Encoder** | Custom ResNet (trained from scratch on Fashion-MNIST/custom data) | **DINO ViT** (pre-trained on millions of images) |
| **Language Grounding** | ConceptBinder (learns 10 class mappings) | **CLIP** (understands any text description) |
| **Recognition Capability** | Only recognizes trained classes | **Zero-shot**: Recognizes ANY concept |
| **Training Required** | Yes (5 epochs SimCLR + 5 epochs binder) | **No** (pure inference) |

---

## New Capabilities

### 1. Zero-Shot Recognition

Upload an image of **anything** and it will recognize it—even if it's never been trained on that concept:

```python
# Recognizes concepts like:
- "a platypus"
- "the Eiffel Tower"
- "abstract expressionism"
- "a quantum computer"
- "medieval armor"
```

### 2. Natural Language Queries

Use free-form text descriptions:

```python
- "something you can eat"
- "a mode of transport"
- "a cute animal"
- "architecture"
- "technology"
```

### 3. Universal Grounding

CLIP maps both images and text to the same vector space, enabling:
- Image-to-text similarity
- Text-to-image search
- Cross-modal reasoning

---

## Quick Start

### 1. Install New Dependencies

```bash
pip install timm open-clip-torch
```

Or update all dependencies:

```bash
pip install -r requirements.txt
```

### 2. Run Tests

Verify that SOTA models work:

```bash
python test_sota.py
```

Expected output:
```
Test Summary
=====================================
Imports                        ✓ PASSED
DINO Encoder                   ✓ PASSED
CLIP Binder                    ✓ PASSED
Zero-Shot Recognition          ✓ PASSED
Cross-Modal Similarity         ✓ PASSED

Total: 5/5 tests passed
```

### 3. Run Demo

See zero-shot recognition in action:

```bash
python src/main_sota.py
```

Or interactive mode:

```bash
python src/main_sota.py --interactive
```

### 4. Launch Dashboard

Interactive web interface:

```bash
streamlit run src/dashboard_sota.py
```

Then open http://localhost:8501 in your browser.

---

## Architecture

### DINO (Self-Distillation with No Labels)

**What is it?**
- A Vision Transformer (ViT) trained with self-supervised learning
- Learns rich visual representations without any labels
- The [CLS] token captures global image semantics

**Why use it?**
- Already understands geometry, depth, and object relationships
- 768-dimensional embeddings capture fine-grained concepts
- Frozen weights = no training needed

**Model:** `vit_base_patch16_224.dino`
- Input: 224×224 RGB images
- Output: 768-D concept vector

### CLIP (Contrastive Language-Image Pre-training)

**What is it?**
- Jointly trained on 400 million (image, text) pairs
- Maps images and text to the same vector space
- Enables zero-shot classification

**Why use it?**
- Universal language-vision grounding
- Understands arbitrary text descriptions
- Robust to novel concepts

**Model:** `ViT-B/32`
- Input: 224×224 images or text strings
- Output: 512-D embeddings (same space for both modalities)

---

## Usage Examples

### Example 1: Classify an Image

```python
from src.world_model.encoder_sota import SOTA_Encoder
from src.language.binder_sota import SOTA_Binder
import torch

# Load models
encoder = SOTA_Encoder(freeze=True)
binder = SOTA_Binder(freeze=True)

# Load image (3 channels, any size)
image = torch.randn(1, 3, 224, 224)

# Define concepts to test
concepts = ["a dog", "a cat", "a car", "a building"]

# Classify
pred_indices, probs = binder.classify(image, concepts)

print(f"Prediction: {concepts[pred_indices[0]]}")
print(f"Confidence: {probs[0, pred_indices[0]]:.2%}")
```

### Example 2: Free-Form Text Query

```python
# Encode image
image_features = binder.embed_image(image)

# Ask arbitrary question
query = "Is this something you can eat?"
query_features = binder.embed_text(query)

# Compute similarity
similarity = (image_features @ query_features.T).item()
match_pct = (similarity + 1) / 2 * 100

print(f"Match: {match_pct:.1f}%")
```

### Example 3: Episodic Memory with SOTA

```python
from src.memory.episodic import EpisodicMemory

memory = EpisodicMemory()

# Store observations
for step, image in enumerate(images):
    features = binder.embed_image(image)
    memory.store(features, step)

# Query by concept
query = "Find memories of animals"
query_vec = binder.embed_text(query)
matches = memory.recall(query_vec, threshold=0.6)

print(f"Found {len(matches)} matching memories")
```

---

## File Structure

```
NSCA/
├── src/
│   ├── world_model/
│   │   ├── encoder.py              # Original custom encoder
│   │   └── encoder_sota.py         # NEW: DINO ViT encoder
│   ├── language/
│   │   ├── binder.py               # Original ConceptBinder
│   │   └── binder_sota.py          # NEW: CLIP binder
│   ├── main.py                     # Original demo (trained models)
│   ├── main_sota.py                # NEW: Zero-shot demo
│   ├── dashboard.py                # Original dashboard
│   └── dashboard_sota.py           # NEW: Zero-shot dashboard
├── requirements.txt                # Updated with timm, open-clip
├── test_sota.py                    # NEW: SOTA test suite
└── SOTA_README.md                  # This file
```

---

## Comparison: Custom vs. SOTA

### Custom Encoder + Binder (Phases 1-12)

**Pros:**
- Full control over architecture
- Learns from limited data
- Interpretable "concept islands"
- Educational (shows how learning works)

**Cons:**
- Requires training (5-10 epochs)
- Limited to trained concepts
- Lower accuracy on novel objects

### SOTA (Phase 14)

**Pros:**
- Zero-shot: Recognizes anything
- No training required
- State-of-the-art accuracy
- Understands natural language

**Cons:**
- Black box (less interpretable)
- Larger models (slower inference)
- Requires internet for first download

---

## Performance Notes

### Model Sizes

- **DINO ViT-Base**: ~86M parameters
- **CLIP ViT-B/32**: ~151M parameters (text + vision)

### Inference Speed

On CPU (M1 Mac):
- DINO: ~50ms per image
- CLIP: ~30ms per text, ~50ms per image

On GPU (CUDA):
- DINO: ~5ms per image
- CLIP: ~3ms per text, ~5ms per image

### Memory

- DINO: ~350MB GPU memory
- CLIP: ~600MB GPU memory

---

## Troubleshooting

### ImportError: No module named 'timm'

Install missing dependencies:

```bash
pip install timm open-clip-torch
```

### Models downloading slowly

Models are cached in `~/.cache/torch/hub/` and `~/.cache/clip/`. First run downloads:
- DINO ViT: ~330MB
- CLIP ViT-B/32: ~350MB

### CUDA out of memory

Reduce batch size or use CPU:

```python
device = "cpu"  # Force CPU
binder = SOTA_Binder(device=device)
```

### Different embedding dimensions

DINO outputs 768-D, CLIP outputs 512-D. For compatibility, use `SOTA_EncoderWithProjection`:

```python
from src.world_model.encoder_sota import SOTA_EncoderWithProjection

encoder = SOTA_EncoderWithProjection(target_dim=512)  # Match CLIP
```

---

## Next Steps

1. **Try the dashboard**: Upload exotic images (platypus, rare cars, unusual architecture)
2. **Experiment with prompts**: See how different text descriptions affect recognition
3. **Combine with memory**: Build a personal image search engine
4. **Fine-tune (optional)**: Unfreeze and adapt to your specific domain

---

## References

- **DINO**: Emerging Properties in Self-Supervised Vision Transformers ([paper](https://arxiv.org/abs/2104.14294))
- **CLIP**: Learning Transferable Visual Models From Natural Language Supervision ([paper](https://arxiv.org/abs/2103.00020))
- **Timm**: PyTorch Image Models ([repo](https://github.com/huggingface/pytorch-image-models))
- **OpenCLIP**: Open source implementation of CLIP ([repo](https://github.com/mlfoundations/open_clip))

---

## License

Same as main project (MIT).
