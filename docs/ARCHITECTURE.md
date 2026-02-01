# NSCA System Architecture

## Overview

The Neuro-Symbolic Cognitive Architecture (NSCA) implements a **five-layer cognitive stack** inspired by biological cognition. Each layer builds upon the previous, creating increasingly abstract and semantically meaningful representations.

## Design Principles

### 1. Innate Priors Over Tabula Rasa

Unlike conventional deep learning that learns everything from pixels, NSCA incorporates **biologically-inspired innate priors**:

- **Color Opponency**: Red-green, blue-yellow opponent channels (retinal ganglion cells)
- **Edge Detection**: Gabor filters mimicking V1 simple cells
- **Depth Cues**: Height-in-field, texture gradients
- **Auditory Processing**: Mel-frequency filterbanks (basilar membrane)
- **Temporal Causality**: Causal masking, exponential decay

### 2. Properties Over Patterns

Objects are understood through **semantic properties** rather than pixel patterns:

```
Traditional: Image → CNN → "rock" (pattern matching)
NSCA: Image → Priors → Properties (hard=0.9, gray, small) → "rock" (grounded)
```

### 3. Causation Over Correlation

Learning emphasizes **causal relationships** through intervention:

```
Correlation: "When A, then B" (statistical)
Causation: "I did A, therefore B" (intervention-based)
```

### 4. Intrinsic Motivation Over External Reward

The agent learns because of **internal drives**, not just external feedback:

- **Curiosity**: Prediction error as reward signal
- **Competence**: Satisfaction from successful predictions
- **Information Gain**: Reduction of uncertainty

---

## Layer 0: World Model

The foundational layer that perceives and predicts the physical world.

### Innate Priors (`src/priors/`)

```
┌─────────────────────────────────────────────────────────────────┐
│                     INNATE PRIORS                               │
├─────────────────────────────────────────────────────────────────┤
│ Visual                                                          │
│ ├─ ColorOpponencyPrior: RGB → L, R-G, B-Y opponent channels    │
│ ├─ GaborPrior: 8 orientations × 4 scales edge detection        │
│ ├─ DepthCuesPrior: Height-in-field → relative depth            │
│ └─ TextureGradientPrior: Local frequency → depth from texture  │
├─────────────────────────────────────────────────────────────────┤
│ Auditory                                                        │
│ ├─ AuditoryPrior: Waveform → Mel spectrogram (cochlear)        │
│ ├─ OnsetDetector: Spectral flux for event detection            │
│ ├─ SpectralContrastPrior: Peak-valley contrast                 │
│ └─ PitchPrior: Autocorrelation-based F0 estimation             │
├─────────────────────────────────────────────────────────────────┤
│ Spatial                                                         │
│ ├─ SpatialPrior3D: 2D rotary embeddings + perspective          │
│ ├─ OcclusionPrior: Depth-based occlusion weighting             │
│ ├─ CenterSurroundPrior: Foveal attention bias                  │
│ └─ GridCellPrior: Periodic position encoding                   │
├─────────────────────────────────────────────────────────────────┤
│ Temporal                                                        │
│ ├─ TemporalPrior: Causal masking + exponential decay           │
│ ├─ RelativeTemporalEncoding: Relative time encoding            │
│ ├─ CausalityPrior: Asymmetric temporal attention               │
│ └─ RhythmPrior: Periodic pattern detection                     │
└─────────────────────────────────────────────────────────────────┘
```

### Multi-Modal Encoders (`src/encoders/`)

Each encoder integrates relevant priors:

```python
# Vision Encoder
VisionEncoderWithPriors:
    Input: RGB [B, C, H, W]
    → ColorOpponencyPrior → opponent channels
    → GaborPrior → edge features  
    → ResNet blocks → latent features
    → SpatialPrior3D → 3D-aware features
    Output: [B, latent_dim]

# Audio Encoder
AudioEncoderWithPriors:
    Input: Waveform [B, samples]
    → AuditoryPrior → mel spectrogram
    → OnsetDetector → onset features (optional)
    → CNN → latent features
    Output: [B, latent_dim]

# Proprioceptive Encoder
ProprioEncoder:
    Input: Body state [B, 12] (position, velocity, acceleration, orientation)
    → MLP → latent features
    Output: [B, latent_dim]
```

### Cross-Modal Fusion (`src/fusion/`)

Attention-based integration of modalities:

```
Vision Features ─┐
                 │
Audio Features  ─┼─→ Modality Embeddings → Concatenate → Self-Attention → Fused
                 │
Proprio Features ┘
```

### Temporal World Model (`src/world_model/temporal_world_model.py`)

Processes sequences with causal structure:

```python
TemporalWorldModel:
    Input: Fused features [B, T, D]
    → TemporalPrior (position encoding, causal mask)
    → Transformer Encoder (causal attention)
    → State aggregation
    Output: World state [B, state_dim]
```

### Dynamics Predictor (`src/world_model/enhanced_dynamics.py`)

Imagination engine for future state prediction:

```python
EnhancedDynamicsPredictor:
    Input: Current state [B, D], Action [B, A]
    → State-action fusion
    → Residual blocks
    → Delta prediction + uncertainty estimation
    Output: Next state [B, D], Uncertainty [B, 1]
```

### Dual Memory (`src/memory/dual_memory.py`)

Two complementary memory systems:

```
┌─────────────────────────────────────────────────────────┐
│ EPISODIC MEMORY                                         │
│ ├─ Stores specific experiences                          │
│ ├─ Vector + metadata + timestamp                        │
│ ├─ Cosine similarity retrieval                          │
│ └─ Recency-weighted access                              │
├─────────────────────────────────────────────────────────┤
│ SEMANTIC MEMORY                                         │
│ ├─ Stores general concepts (prototypes)                 │
│ ├─ Learned from repeated patterns                       │
│ ├─ Prototype + count + variance                         │
│ └─ Consolidation: episodic → semantic                   │
└─────────────────────────────────────────────────────────┘
```

---

## Layer 1: Semantic Properties

Extracts meaningful properties from world state.

### Property Layer (`src/semantics/property_layer.py`)

```
┌─────────────────────────────────────────────────────────────────┐
│ PROPERTY EXTRACTION                                             │
├─────────────────────────────────────────────────────────────────┤
│ Property          │ Source                │ Range               │
├───────────────────┼───────────────────────┼─────────────────────┤
│ Hardness          │ Audio (high freq)     │ [0=soft, 1=hard]    │
│ Weight            │ Proprio (force/accel) │ [0=light, 1=heavy]  │
│ Size              │ Visual (extent)       │ [0=tiny, 1=large]   │
│ Animacy           │ Motion (self-propel)  │ [0=inanimate, 1=animate] │
│ Rigidity          │ Visual (deformation)  │ [0=flexible, 1=rigid] │
│ Transparency      │ Visual                │ [0=opaque, 1=clear] │
│ Roughness         │ Visual (texture)      │ [0=smooth, 1=rough] │
│ Temperature       │ Context-inferred      │ [0=cold, 1=hot]     │
│ Containment       │ Visual (shape)        │ [0=solid, 1=hollow] │
└─────────────────────────────────────────────────────────────────┘
```

### Affordances (`src/semantics/affordances.py`)

What actions are possible with objects:

```
Graspable:   Small + light + not animate
Sittable:    Large + stable + rigid
Throwable:   Small + light + graspable
Containable: Hollow + rigid
Breakable:   Rigid + not too hard
```

### Categories (`src/semantics/categories.py`)

Fundamental ontological types:

- **Agent**: Self-propelled, goal-directed
- **Object**: Passive, bounded
- **Substance**: Continuous (water, sand)
- **Container**: Has interior space
- **Surface**: Flat, supports things
- **Tool**: Used for manipulation

---

## Layer 2: Causal Reasoning

Understands WHY things happen.

### Causal Reasoner (`src/reasoning/causal_layer.py`)

```
┌─────────────────────────────────────────────────────────────────┐
│ CAUSAL LEARNING                                                 │
├─────────────────────────────────────────────────────────────────┤
│ Type          │ Description                                     │
├───────────────┼─────────────────────────────────────────────────┤
│ Intervention  │ "I did X, then Y happened" → I caused Y        │
│ Observation   │ "X happened, then Y" → X may cause Y           │
│ Physics       │ Gravity, collision → physical laws caused it   │
│ Agent         │ Another agent's action caused it               │
└─────────────────────────────────────────────────────────────────┘
```

### Intuitive Physics (`src/reasoning/intuitive_physics.py`)

Partially innate physics expectations:

```python
GravityPrior:     Unsupported objects fall
SolidityPrior:    Objects don't pass through each other
ContactCausality: Causation requires contact (usually)
SupportPrior:     Objects need support to stay up
```

### Counterfactual Reasoning (`src/reasoning/counterfactual.py`)

"What if" scenarios:

```
Actual:       I pushed the ball → it rolled
Counterfactual: If I hadn't pushed → would it have rolled?
Conclusion:   My push CAUSED the rolling
```

---

## Layer 3: Motivation / Drives

Provides intrinsic reasons to learn.

### Drive System (`src/motivation/drive_system.py`)

```
┌─────────────────────────────────────────────────────────────────┐
│ INNATE DRIVES                                                   │
├─────────────────────────────────────────────────────────────────┤
│ Drive        │ Purpose                │ Signal                  │
├──────────────┼────────────────────────┼─────────────────────────┤
│ Curiosity    │ Seek learning          │ Prediction error        │
│ Competence   │ Seek mastery           │ Successful predictions  │
│ Energy       │ Maintain homeostasis   │ Activity level          │
│ Safety       │ Avoid danger           │ Threat detection        │
└─────────────────────────────────────────────────────────────────┘
```

### Intrinsic Reward (`src/motivation/intrinsic_reward.py`)

```python
Total Reward = (
    0.4 × Curiosity Reward (prediction error) +
    0.4 × Competence Reward (learning progress) +
    0.2 × Information Gain (uncertainty reduction)
)
```

### Attention Allocation (`src/motivation/attention.py`)

What to focus on:

```
Attention = w₁ × Salience (automatic) +
            w₂ × Relevance (goal-directed) +
            w₃ × Novelty (curiosity) +
            w₄ × Threat (survival)
```

---

## Layer 4: Language Integration

Grounds language in perceptual concepts.

### Language Grounding (`src/language/llm_integration.py`)

```
┌─────────────────────────────────────────────────────────────────┐
│ BIDIRECTIONAL GROUNDING                                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Concept → Language:                                            │
│    Properties [0.9, 0.7, 0.3, 0.0, ...] → "hard, heavy, small" │
│                                                                 │
│  Language → Concept:                                            │
│    "rock" → Properties [0.9, 0.7, 0.3, 0.0, 0.9, 0.0, 0.7...] │
│                                                                 │
│  LLM Integration:                                               │
│    Perceptual description → LLM → Linguistic reasoning          │
│    LLM response → Ground back in concept space                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Unified Cognitive Agent

`src/cognitive_agent.py` integrates all layers:

```python
class CognitiveAgent:
    """
    Complete cognitive architecture.
    
    Methods:
        perceive(vision, audio, proprio) → Full perception through all layers
        act(action) → Process action and update causal understanding
        imagine(actions) → Predict future states
        why_did_this_happen() → Causal explanation
        what_is_this() → Property-based description
        answer_question(question) → Language-grounded response
        remember(label) → Store in memory
        recall(query) → Retrieve from memory
    """
```

---

## Data Flow

```
                          Raw Sensory Input
                                │
                    ┌───────────┼───────────┐
                    │           │           │
                  Vision      Audio      Proprio
                    │           │           │
                    ▼           ▼           ▼
            ┌───────────────────────────────────┐
            │     LAYER 0: WORLD MODEL          │
            │  Priors → Encoders → Fusion →     │
            │  Temporal → Dynamics → Memory     │
            └───────────────┬───────────────────┘
                            │ World State
                            ▼
            ┌───────────────────────────────────┐
            │   LAYER 1: SEMANTIC PROPERTIES    │
            │  Properties → Affordances →       │
            │  Categories                       │
            └───────────────┬───────────────────┘
                            │ Property Vector
                            ▼
            ┌───────────────────────────────────┐
            │    LAYER 2: CAUSAL REASONING      │
            │  Causal Graph → Physics →         │
            │  Counterfactual                   │
            └───────────────┬───────────────────┘
                            │ Causal Understanding
                            ▼
            ┌───────────────────────────────────┐
            │    LAYER 3: MOTIVATION            │
            │  Drives → Intrinsic Reward →      │
            │  Attention                        │
            └───────────────┬───────────────────┘
                            │ Motivation Signal
                            ▼
            ┌───────────────────────────────────┐
            │    LAYER 4: LANGUAGE              │
            │  Verbalize → Ground →             │
            │  LLM Reasoning                    │
            └───────────────┬───────────────────┘
                            │
                            ▼
                    Understanding + Action
```

---

## Configuration

All components are configurable via dataclasses:

```python
from src.cognitive_agent import CognitiveConfig, create_cognitive_agent

config = CognitiveConfig()

# Customize Layer 0
config.world_model.latent_dim = 256
config.world_model.state_dim = 128

# Customize Layer 1
config.property_layer.hidden_dim = 512

# Customize Layer 2
config.causal.num_causal_factors = 32

# Customize Layer 3
config.drive.curiosity_decay = 0.01

# Customize Layer 4
config.language.use_external_llm = True

agent = create_cognitive_agent(config.world_model, use_llm=True)
```

---

## Memory Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    MEMORY SYSTEMS                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Working Memory (current state)                                 │
│  ├─ current_state: Current world representation                 │
│  ├─ previous_state: Last world state (for change detection)    │
│  └─ current_properties: Current property vector                 │
│                                                                 │
│  Episodic Memory (experiences)                                  │
│  ├─ Vector store with metadata                                  │
│  ├─ Cosine similarity retrieval                                 │
│  └─ Max 10,000 entries                                          │
│                                                                 │
│  Semantic Memory (concepts)                                     │
│  ├─ Prototype vectors                                           │
│  ├─ Learned from repeated patterns                              │
│  └─ Consolidated from episodic                                  │
│                                                                 │
│  Causal Memory (relationships)                                  │
│  ├─ Intervention → Effect mappings                              │
│  └─ Causal graph structure                                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## References

- LeCun, Y. (2022). A path towards autonomous machine intelligence.
- Spelke, E. S. (2007). Core knowledge. Developmental Science.
- Kahneman, D. (2011). Thinking, Fast and Slow.
- Lake, B. M., et al. (2017). Building machines that learn and think like people.
- Gopnik, A. (2012). Scientific thinking in young children.
