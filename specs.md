# Neuro-Symbolic JEPA Core — Architecture Specification

## 1. Core Philosophy & Constraints

- **No "Black Boxes"**: We do not rely on massive pre-trained APIs for reasoning. We build the reasoning engine from scratch using PyTorch.
- **Grounding First**: The system must learn "concepts" (vectors) from raw data (pixels/sensors) before attaching language labels.
- **Modularity**: The "Eyes" (Encoder), "Brain" (Predictor), and "Hands" (Tools) must be decoupled.
- **Low-Level Purity**: Use pure `torch.nn` modules. Avoid high-level abstractions like LangChain.

## 2. The Architecture Stack

### Layer 1: The Innate World Model (The "Canvas")

- **Role**: Maintains a latent representation of the world state.
- **Architecture**: Joint-Embedding Predictive Architecture (JEPA).
- **Components**:
  - **SensoryEncoder**: ConvNet/Transformer to map Observation \(x\) → Latent \(z\).
  - **DynamicsPredictor**: Maps \((z_t, \text{action}) \rightarrow z_{t+1}\). The "Imagination Engine."
  - **GeometricBias**: A custom layer enforcing 3D/2D consistency (e.g., rotary embeddings or graph constraints).

### Layer 2: The Manager (The "System 2" Brain)

- **Role**: The decision-making loop.
- **Loop**: Observe → Imagine Consequences → Consult Tools (if uncertain) → Act.
- **Logic**:
  - If prediction uncertainty is **LOW** → Act instinctively.
  - If prediction uncertainty is **HIGH** → Call "Dictionary" (API).

### Layer 3: The Tool Interface (The "Dictionary")

- **Role**: Deterministic lookup for facts the model shouldn't memorize.
- **Implementation**: A standardized `ToolProtocol` interface for connecting APIs (Wikipedia, Calculator, OS).

## 3. Directory Structure

```
project_root/
├── src/
│   ├── world_model/
│   │   ├── __init__.py
│   │   ├── encoder.py       # Sensory System
│   │   ├── geometry.py      # Innate Physics/Priors
│   │   ├── jepa_core.py     # The Learning Algorithm
│   │   └── dynamics.py      # The Predictor/Imagination
│   ├── manager/
│   │   ├── __init__.py
│   │   ├── agent.py         # The Main Loop
│   │   └── memory.py       # Short-term context buffer
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── base.py          # Abstract Base Class for Tools
│   │   └── registry.py     # Tool Dispatcher
│   └── main.py              # Entry point
├── tests/
├── requirements.txt
└── specs.md                 # This file
```

## 4. Implementation Details

### Phase 1: The Sensory Encoder & Geometric Priors

- **File: `src/world_model/encoder.py`**  
  Create a `SpatialEncoder` class using standard CNN blocks (ResNet-style) or a micro-ViT.  
  **CRITICAL**: The output must not be a flat vector. It must be a **SpatialMap** (e.g., tensor shape `[B, C, H, W]`) to preserve spatial relationships.

- **File: `src/world_model/geometry.py`**  
  Implement `RotaryEmbedding2D` or a similar mechanism to encode "position" as a fundamental truth, not a learned feature.  
  This ensures the model knows that "Object A at (0,0)" and "Object A at (1,1)" are the same object, just moved.

### Phase 2: The JEPA Core (The "Imagination")

- **File: `src/world_model/jepa_core.py`**  
  Implement `JEPA_Model` class.  
  Input: Current State (\(z_t\)) + Action (\(a\)).  
  Output: Predicted Next State (\(\hat{z}_{t+1}\)).  
  Loss: VICReg or InfoNCE (not MSE on pixels).

### Phase 3: The Manager Loop

- **File: `src/manager/agent.py`**  
  Implement `CognitiveCycle`.  
  `step(observation)`: encode → measure uncertainty → if high, reason + use tools; else act via policy.

### Phase 4: Tool Protocol

- **File: `src/tools/base.py`**  
  Define a strict Protocol: `name`, `trigger_embedding`, `execute(args)`.

## 5. Coding Standards (Strict Mode)

- **Type Hints**: Every function must have Python 3.10+ type hints.
- **Docstrings**: Every class must explain the physical intuition behind it.
- **No Magic Numbers**: All constants (dimensions, thresholds) in a config dataclass.
- **PyTorch Only**: Do not import `transformers` or `langchain` in the core. Keep the core dependency-free.
