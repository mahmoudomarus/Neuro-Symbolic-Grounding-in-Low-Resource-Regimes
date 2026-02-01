# NeuroSymbolic-JEPA-Core Architecture

*Auto-generated on 2026-01-31 16:14:12*

## System Overview

This project implements a **Neuro-Symbolic Agent** with a Joint-Embedding Predictive Architecture (JEPA) core. The system grounds language in physical concepts learned from raw sensory data, using a modular System 1/System 2 architecture.

### Core Philosophy

1. **No Black Boxes**: The reasoning engine is built from scratch using PyTorch
2. **Grounding First**: Concepts (vectors) are learned from raw data before attaching language labels
3. **Modularity**: Eyes (Encoder), Brain (Predictor), and Hands (Tools) are decoupled
4. **Low-Level Purity**: Pure `torch.nn` modules with no high-level abstractions

## Project Structure

```
project_root/
├── src/
│   ├── world_model/
│   │   ├── config.py       # Configuration dataclasses
│   │   ├── encoder.py      # SpatialEncoder (the "Eyes")
│   │   ├── geometry.py     # Rotary embeddings (spatial priors)
│   │   ├── dynamics.py     # DynamicsPredictor (imagination)
│   │   ├── jepa_core.py    # JEPA model and loss
│   │   └── train_encoder.py # SimCLR training script
│   ├── language/
│   │   ├── binder.py       # ConceptBinder (language grounding)
│   │   └── train_grounding.py # Binder training script
│   ├── memory/
│   │   └── episodic.py     # EpisodicMemory (latent storage)
│   ├── manager/
│   │   ├── agent.py        # CognitiveAgent (System 1/2 loop)
│   │   └── config.py       # Agent configuration
│   ├── tools/
│   │   ├── base.py         # Tool abstract base class
│   │   └── library.py      # Concrete tools (Wiki, Calculator)
│   ├── data/
│   │   └── custom_loader.py # RGB dataset loader
│   ├── dashboard.py        # Streamlit UI
│   └── main.py             # Entry point
├── checkpoints/            # Saved model weights
├── docs/                   # Documentation
├── paper/                  # Academic manuscript
├── scripts/                # Utility scripts
└── tests/                  # Test suite
```

---

## The World Model (JEPA)

The World Model maintains a latent representation of the environment and predicts future states.

### SpatialEncoder (The "Eyes")

**Module:** `src.world_model.encoder`

Encodes visual (or sensory) patches into a latent spatial map.

Uses a CNN backbone (ResNet-style) so the output is a SpatialMap [B, C, H, W],
not a flat vector—preserving where things are. Then applies a geometric prior
(RotaryEmbedding2D) so position is encoded as a fundamental truth: the same
object at (0,0) vs (1,1) is the same concept, just moved.

**Constructor:** `SpatialEncoder(encoder_config: EncoderConfig, geometry_config: Optional[GeometryConfig])`

**Key Methods:**
- `forward(self, x: 'torch.Tensor') -> 'torch.Tensor'`: Map observation to spatial latent map.

Args:
    x: Observation tensor (B, input_channels, H, W), e...
- `output_shape(self, input_height: 'int', input_width: 'int') -> 'Tuple[int, int, int]'`: Return (C, H_out, W_out) for the given input spatial size (for downstream
modules that need to know ...

### DynamicsPredictor (The "Imagination")

**Module:** `src.world_model.dynamics`

Predicts next latent state z_{t+1} from current latent map z_t and action.

Fuses z_t [B, C, H, W] with action [B, A_dim] by broadcasting action to
[B, A_dim, H, W] and concatenating, then runs a stack of ResBlocks to produce
a predicted latent map [B, C, H, W]. Spatial dimensions are never flattened.

**Constructor:** `DynamicsPredictor(config: DynamicsConfig)`

### JEPA Model

**Module:** `src.world_model.jepa_core`

Joint-Embedding Predictive Architecture: encoder + dynamics predictor.

Encodes current frame x_t (with grad) and next frame x_{t+1} (target, no grad).
Predicts z_{t+1} from z_t and action. Loss = consistency (MSE) + variance
regularization to prevent collapse.

---

## The Concept Binder (Language Grounding)

**Module:** `src.language.binder`

Maps class indices to normalized vectors in the encoder's latent space.

Uses a simple embedding table so that label k (e.g. 7 = Sneaker) maps to a
vector that can be aligned with pooled visual features via cosine similarity.
Output dim must match encoder output_channels (pooled).

**Constructor:** `ConceptBinder(num_classes: int, embedding_dim: int)`

The Concept Binder maps class indices (e.g., 7 = "Sneaker") to normalized vectors in the same latent space as the encoder output. This enables:
- **Language → Vision**: Given a word, "imagine" its visual form
- **Vision → Language**: Given an image, find the closest concept name

---

## Episodic Memory

**Module:** `src.memory.episodic`

Stores a sequence of experiences as normalized latent vectors.
Recall returns steps where stored vectors match a query (cosine similarity).

**Key Methods:**
- `clear(self) -> 'None'`: Reset memory (empty storage).
- `get_recent(self, n: 'int' = 10) -> 'list[dict[str, Any]]'`: Return the last n entries (for display). Each entry has keys: timestamp, meta, vector.
- `recall(self, query_vector: 'torch.Tensor', threshold: 'float' = 0.8) -> 'list[tuple[int, float]]'`: Compute cosine similarity of query against all stored vectors.
Returns list of (step, similarity) fo...
- `store(self, observation_vector: 'torch.Tensor', step: 'int', metadata: 'Any' = None) -> 'None'`: Store a pooled latent vector for this step.
Detaches and moves to CPU to save memory; normalizes bef...

The memory operates entirely in **latent space** — raw images are never stored. This enables:
- Efficient similarity search (cosine similarity on 64-dim vectors)
- Privacy-preserving storage (original pixels are not recoverable)
- Concept-based recall ("Find memories similar to 'Sneaker'")

---

## The Manager (System 1/2 Loop)

**Module:** `src.manager.agent`

Agent that either acts directly (System 1) or calls a tool (System 2) based
on latent uncertainty. Uses a heuristic for uncertainty; can be replaced
later by a learned head.

### Decision Flow

```
Observation → Encoder → Latent State → Measure Uncertainty
                                              │
                              ┌───────────────┴───────────────┐
                              │                               │
                        Low Uncertainty               High Uncertainty
                        (System 1)                    (System 2)
                              │                               │
                        Direct Action              Tool Selection
                        "ACTION: MOVE"             (Cosine Similarity)
                                                          │
                                                  "TOOL_CALL: Wiki"
```

---

## Configuration

### EncoderConfig

| Field | Type | Description |
|-------|------|-------------|
| `input_channels` | `int` | Default: factory |
| `base_channels` | `int` | Default: factory |
| `num_blocks` | `int` | Default: factory |
| `output_channels` | `int` | Default: factory |
| `strides_per_stage` | `Tuple` | Default: factory |
| `use_geometry` | `bool` | Default: True |

### RealWorldConfig (RGB Mode)

| Field | Type | Description |
|-------|------|-------------|
| `input_resolution` | `int` | Default: 64 |
| `input_channels` | `int` | Default: 3 |
| `patch_size` | `int` | Default: 8 |
| `normalize_mean` | `Tuple` | Default: (0.485, 0.456, 0.406) |
| `normalize_std` | `Tuple` | Default: (0.229, 0.224, 0.225) |

### AgentConfig

| Field | Type | Description |
|-------|------|-------------|
| `uncertainty_threshold` | `float` | Default: 0.7 |

---

## Training Pipeline

### Phase 1: Vision Training (SimCLR)

The `SpatialEncoder` is trained using **contrastive learning** (SimCLR):
1. Each image is augmented twice to create positive pairs
2. NT-Xent loss maximizes similarity between positive pairs
3. Encoder learns to map similar concepts to nearby vectors

### Phase 2: Language Grounding

The `ConceptBinder` is trained to align word vectors with visual vectors:
1. Encoder is frozen (from Phase 1)
2. For each (image, label) pair, maximize cosine similarity between:
   - `encoder(image)` (pooled)
   - `binder(label)`
3. After training, `binder(7)` ≈ `encoder(sneaker_image)`

---

## Data Flow

```
                    ┌─────────────────────────────────────────────┐
                    │              Custom RGB Dataset              │
                    │         ./data/my_dataset/class_*/          │
                    └─────────────────┬───────────────────────────┘
                                      │
                    ┌─────────────────▼───────────────────────────┐
                    │            SimCLR Training                   │
                    │     (train_encoder.py)                       │
                    └─────────────────┬───────────────────────────┘
                                      │
                    ┌─────────────────▼───────────────────────────┐
                    │           encoder_v1.pth                     │
                    └─────────────────┬───────────────────────────┘
                                      │
                    ┌─────────────────▼───────────────────────────┐
                    │         Language Grounding                   │
                    │     (train_grounding.py)                     │
                    └─────────────────┬───────────────────────────┘
                                      │
              ┌───────────────────────┼───────────────────────────┐
              │                       │                           │
   ┌──────────▼──────────┐  ┌────────▼────────┐  ┌───────────────▼────────────┐
   │   binder_v1.pth     │  │ dataset_config  │  │        Dashboard           │
   │                     │  │     .json       │  │   (streamlit dashboard)    │
   └─────────────────────┘  └─────────────────┘  └────────────────────────────┘
```

---

## License

MIT License - See LICENSE file for details.
