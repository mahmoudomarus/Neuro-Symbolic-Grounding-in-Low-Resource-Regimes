# NSCA System Architecture

## Overview

The Neuro-Symbolic Cognitive Architecture (NSCA) implements a **five-layer cognitive stack** inspired by biological cognition. Each layer builds upon the previous, creating increasingly abstract and semantically meaningful representations.

**Architectural Update (v2.0)**: Following peer review, the architecture now implements **adaptive priors** rather than fixed rules, **learned grounding** through sensorimotor babbling, and **robust intrinsic motivation** with noisy-TV defense.

## Design Principles

### 1. Adaptive Priors Over Fixed Rules

**Key Innovation**: Priors are now *learnable biases*, not hard-coded rules.

```
OLD: prior = fixed_value (breaks on exceptions)
NEW: output = w × prior + (1-w) × correction (learns exceptions)
```

The **critical period floor** (w ≥ 0.3) ensures physics knowledge is never completely forgotten:

```python
effective_weight = 0.3 + softplus(learned_weight - 0.3)  # Always ≥ 0.3
```

### 2. Learned Grounding Over Hard-Coded Dictionaries

**Removed**: All manual concept groundings (CONCEPT_GROUNDINGS dict)
**Added**: Sensorimotor babbling protocol

```
OLD: "rock" → [0.9, 0.7, 0.3, ...] (manually defined)
NEW: "rock" → strike_interaction → audio_frequency → learned hardness
```

### 3. Properties Over Patterns (with Dynamic Slots)

Objects are understood through **semantic properties**, now with open-ended discovery:

```
Traditional: Image → CNN → "rock" (pattern matching)
NSCA v1: Image → Priors → 9 fixed properties → "rock"
NSCA v2: Image → Priors → Slot Attention → 9 known + N discoverable properties
```

### 4. Robust Intrinsic Motivation

**Problem Solved**: "Noisy TV" problem (random noise has high prediction error)
**Solution**: Learnability filtering

```python
reward = prediction_error × learnability
# High error + high learnability = learn (novel physics)
# High error + low learnability = ignore (random noise)
```

### 5. Continual Learning with EWC

**Added**: Elastic Weight Consolidation prevents catastrophic forgetting
- Semantic memory: 10× protection (consolidated knowledge)
- Episodic memory: 1× protection (can be overwritten)

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

**UPDATED**: Now includes `DynamicPropertyBank` for open-ended property discovery:

```
┌─────────────────────────────────────────────────────────────────┐
│ DYNAMIC PROPERTY EXTRACTION (Slot Attention)                    │
├─────────────────────────────────────────────────────────────────┤
│ KNOWN SLOTS (0-8) - Initialized with priors                     │
├───────────────────┼───────────────────────┼─────────────────────┤
│ Slot 0: Hardness  │ Audio (high freq)     │ [0=soft, 1=hard]    │
│ Slot 1: Weight    │ Video dynamics        │ [0=light, 1=heavy]  │
│ Slot 2: Size      │ Visual (extent)       │ [0=tiny, 1=large]   │
│ Slot 3: Animacy   │ Motion (self-propel)  │ [0=inanimate, 1=animate] │
│ Slot 4: Rigidity  │ Visual (deformation)  │ [0=flexible, 1=rigid] │
│ Slot 5: Transparency │ Visual             │ [0=opaque, 1=clear] │
│ Slot 6: Roughness │ Visual (texture)      │ [0=smooth, 1=rough] │
│ Slot 7: Temperature │ Context-inferred    │ [0=cold, 1=hot]     │
│ Slot 8: Containment │ Visual (shape)      │ [0=solid, 1=hollow] │
├─────────────────────────────────────────────────────────────────┤
│ FREE SLOTS (9-31) - Activated by prediction error               │
├───────────────────┼───────────────────────────────────────────────┤
│ Slot 9+: ???      │ Discovered through learning                  │
│                   │ E.g., "stickiness", "elasticity"             │
│                   │ Grounded post-hoc via LLM/human labeling     │
└─────────────────────────────────────────────────────────────────┘
```

**Also added**: `RobustSlotAttention` with reconstruction-based OOD detection
- If reconstruction error > threshold, output "uncertain"
- Prevents adversarial inputs from populating symbolic layer

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

**UPDATED**: Now uses **Adaptive Physics Priors** with residual correction networks:

```python
# Old (brittle)
GravityPrior:     if not supported: fall  # Breaks on balloons

# New (adaptive)
AdaptivePhysicsPrior:
    prior_motion = [0, -9.8, 0]  # Innate gravity expectation
    correction = correction_net(object_state)  # Learned exceptions
    output = w × prior_motion + (1-w) × correction
    
    # Critical period: w always ≥ 0.3 (never forgets physics)
    # Balloon training: w drops to ~0.35, correction learns "up"
```

Physics laws with adaptive priors:
- **Gravity**: Objects fall (but balloons can override)
- **Solidity**: Objects don't pass through (but permeable surfaces exist)
- **Contact**: Causation requires contact (but magnets don't)
- **Support**: Objects need support (but levitation possible)

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

**UPDATED**: Now uses `RobustCuriosityReward` with noisy-TV defense:

```python
# Old (vulnerable to noisy TV)
curiosity_reward = prediction_error  # Random noise gets high reward!

# New (RobustCuriosityReward)
learnability = (early_error - recent_error) / early_error  # Error reduction
curiosity_reward = prediction_error × learnability

# Noisy TV: High error but no decrease → learnability ≈ 0 → reward ≈ 0
# Novel physics: High error, decreases over time → high reward

Total Reward = (
    0.4 × RobustCuriosityReward +  # With EMA encoder for stable hashing
    0.4 × Competence Reward +
    0.2 × Information Gain
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

**CRITICAL UPDATE**: Removed all hard-coded concept groundings. Grounding now learned via babbling.

```
┌─────────────────────────────────────────────────────────────────┐
│ LEARNED GROUNDING (via Sensorimotor Babbling)                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  OLD (REMOVED - was scientific misconduct):                     │
│    CONCEPT_GROUNDINGS = {"rock": [0.9, 0.7, ...]}              │
│                                                                 │
│  NEW (LearnedGrounding):                                        │
│    grounding_table = {}  # Starts EMPTY                         │
│                                                                 │
│  Babbling Protocol:                                             │
│    Phase 1 (1000 steps): Random exploration                     │
│    Phase 2 (9000 steps): Competence-driven (retry learnable)    │
│                                                                 │
│  Learning from Interaction:                                     │
│    strike(rock) → audio_frequency=0.9 → grounding["rock"][0]=0.9│
│    lift(rock) → force_required=0.7 → grounding["rock"][1]=0.7  │
│                                                                 │
│  Evaluation: Report "zero-shot AFTER babbling", not "zero-shot" │
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

**UPDATED**: Added Elastic Weight Consolidation (EWC) to prevent catastrophic forgetting.

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
│  Episodic Memory (experiences) - 1× EWC protection              │
│  ├─ Vector store with metadata                                  │
│  ├─ Cosine similarity retrieval                                 │
│  └─ Max 10,000 entries (can be overwritten)                     │
│                                                                 │
│  Semantic Memory (concepts) - 10× EWC protection                │
│  ├─ Prototype vectors                                           │
│  ├─ Learned from repeated patterns                              │
│  ├─ Consolidated from episodic                                  │
│  └─ HIGHLY PROTECTED (consolidated knowledge)                   │
│                                                                 │
│  Causal Memory (relationships)                                  │
│  ├─ Intervention → Effect mappings                              │
│  └─ Causal graph structure                                      │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│  EWC Protection (`src/learning/ewc.py`)                         │
│  ├─ Computes Fisher information (parameter importance)          │
│  ├─ Penalty for changing important weights                      │
│  └─ 10× multiplier for semantic memory parameters               │
└─────────────────────────────────────────────────────────────────┘
```

---

## References

- LeCun, Y. (2022). A path towards autonomous machine intelligence.
- Spelke, E. S. (2007). Core knowledge. Developmental Science.
- Kahneman, D. (2011). Thinking, Fast and Slow.
- Lake, B. M., et al. (2017). Building machines that learn and think like people.
- Gopnik, A. (2012). Scientific thinking in young children.
