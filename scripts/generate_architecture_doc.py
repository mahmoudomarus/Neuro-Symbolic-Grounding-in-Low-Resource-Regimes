#!/usr/bin/env python3
"""
Auto-generate ARCHITECTURE.md from code introspection.

This script introspects the main classes and modules of the NeuroSymbolic-JEPA-Core
project and generates structured documentation in docs/ARCHITECTURE.md.

Run from project root: python scripts/generate_architecture_doc.py
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# Add project root to path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


def get_class_info(cls: type) -> Dict[str, Any]:
    """Extract class information including docstring and method signatures."""
    info = {
        "name": cls.__name__,
        "module": cls.__module__,
        "docstring": inspect.getdoc(cls) or "No documentation available.",
        "methods": [],
    }
    
    # Get __init__ signature
    try:
        init_sig = inspect.signature(cls.__init__)
        params = []
        for name, p in init_sig.parameters.items():
            if name == "self":
                continue
            if p.annotation == inspect.Parameter.empty:
                type_str = "Any"
            elif hasattr(p.annotation, "__name__"):
                type_str = p.annotation.__name__
            else:
                type_str = str(p.annotation)
            params.append(f"{name}: {type_str}")
        info["init_params"] = ", ".join(params) if params else "none"
    except (ValueError, TypeError):
        info["init_params"] = "unknown"
    
    # Get public methods
    for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
        if not name.startswith("_") or name == "__init__":
            try:
                sig = inspect.signature(method)
                method_doc = inspect.getdoc(method)
                info["methods"].append({
                    "name": name,
                    "signature": str(sig),
                    "docstring": method_doc[:100] + "..." if method_doc and len(method_doc) > 100 else method_doc,
                })
            except (ValueError, TypeError):
                pass
    
    return info


def get_dataclass_info(cls: type) -> Dict[str, Any]:
    """Extract dataclass field information."""
    info = {
        "name": cls.__name__,
        "module": cls.__module__,
        "docstring": inspect.getdoc(cls) or "No documentation available.",
        "fields": [],
    }
    
    # Get dataclass fields
    if hasattr(cls, "__dataclass_fields__"):
        for name, field in cls.__dataclass_fields__.items():
            field_type = field.type.__name__ if hasattr(field.type, "__name__") else str(field.type)
            default = field.default if field.default is not field.default_factory else "factory"
            info["fields"].append({
                "name": name,
                "type": field_type,
                "default": str(default) if default != inspect._empty else "required",
            })
    
    return info


def generate_project_tree() -> str:
    """Generate a simplified project directory tree."""
    tree = """```
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
```"""
    return tree


def generate_architecture_doc() -> str:
    """Generate the full architecture documentation."""
    
    # Import modules for introspection
    from src.world_model.encoder import SpatialEncoder
    from src.world_model.dynamics import DynamicsPredictor
    from src.world_model.jepa_core import JEPA
    from src.world_model.config import (
        EncoderConfig, DynamicsConfig, JEPAConfig, TrainingConfig, RealWorldConfig
    )
    from src.language.binder import ConceptBinder
    from src.memory.episodic import EpisodicMemory
    from src.manager.agent import CognitiveAgent
    from src.manager.config import AgentConfig
    from src.tools.base import Tool
    
    # Gather class information
    encoder_info = get_class_info(SpatialEncoder)
    dynamics_info = get_class_info(DynamicsPredictor)
    jepa_info = get_class_info(JEPA)
    binder_info = get_class_info(ConceptBinder)
    memory_info = get_class_info(EpisodicMemory)
    agent_info = get_class_info(CognitiveAgent)
    
    # Gather config information
    encoder_config_info = get_dataclass_info(EncoderConfig)
    dynamics_config_info = get_dataclass_info(DynamicsConfig)
    realworld_config_info = get_dataclass_info(RealWorldConfig)
    agent_config_info = get_dataclass_info(AgentConfig)
    
    doc = f"""# NeuroSymbolic-JEPA-Core Architecture

*Auto-generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*

## System Overview

This project implements a **Neuro-Symbolic Agent** with a Joint-Embedding Predictive Architecture (JEPA) core. The system grounds language in physical concepts learned from raw sensory data, using a modular System 1/System 2 architecture.

### Core Philosophy

1. **No Black Boxes**: The reasoning engine is built from scratch using PyTorch
2. **Grounding First**: Concepts (vectors) are learned from raw data before attaching language labels
3. **Modularity**: Eyes (Encoder), Brain (Predictor), and Hands (Tools) are decoupled
4. **Low-Level Purity**: Pure `torch.nn` modules with no high-level abstractions

## Project Structure

{generate_project_tree()}

---

## The World Model (JEPA)

The World Model maintains a latent representation of the environment and predicts future states.

### SpatialEncoder (The "Eyes")

**Module:** `{encoder_info['module']}`

{encoder_info['docstring']}

**Constructor:** `SpatialEncoder({encoder_info['init_params']})`

**Key Methods:**
"""
    
    for method in encoder_info["methods"]:
        if method["name"] in ["forward", "output_shape"]:
            doc += f"- `{method['name']}{method['signature']}`: {method['docstring'] or 'See source.'}\n"
    
    doc += f"""
### DynamicsPredictor (The "Imagination")

**Module:** `{dynamics_info['module']}`

{dynamics_info['docstring']}

**Constructor:** `DynamicsPredictor({dynamics_info['init_params']})`

### JEPA Model

**Module:** `{jepa_info['module']}`

{jepa_info['docstring']}

---

## The Concept Binder (Language Grounding)

**Module:** `{binder_info['module']}`

{binder_info['docstring']}

**Constructor:** `ConceptBinder({binder_info['init_params']})`

The Concept Binder maps class indices (e.g., 7 = "Sneaker") to normalized vectors in the same latent space as the encoder output. This enables:
- **Language → Vision**: Given a word, "imagine" its visual form
- **Vision → Language**: Given an image, find the closest concept name

---

## Episodic Memory

**Module:** `{memory_info['module']}`

{memory_info['docstring']}

**Key Methods:**
"""
    
    for method in memory_info["methods"]:
        if method["name"] in ["store", "recall", "clear", "get_recent"]:
            doc += f"- `{method['name']}{method['signature']}`: {method['docstring'] or 'See source.'}\n"
    
    doc += f"""
The memory operates entirely in **latent space** — raw images are never stored. This enables:
- Efficient similarity search (cosine similarity on 64-dim vectors)
- Privacy-preserving storage (original pixels are not recoverable)
- Concept-based recall ("Find memories similar to 'Sneaker'")

---

## The Manager (System 1/2 Loop)

**Module:** `{agent_info['module']}`

{agent_info['docstring']}

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
"""
    
    for field in encoder_config_info["fields"]:
        doc += f"| `{field['name']}` | `{field['type']}` | Default: {field['default']} |\n"
    
    doc += f"""
### RealWorldConfig (RGB Mode)

| Field | Type | Description |
|-------|------|-------------|
"""
    
    for field in realworld_config_info["fields"]:
        doc += f"| `{field['name']}` | `{field['type']}` | Default: {field['default']} |\n"
    
    doc += f"""
### AgentConfig

| Field | Type | Description |
|-------|------|-------------|
"""
    
    for field in agent_config_info["fields"]:
        doc += f"| `{field['name']}` | `{field['type']}` | Default: {field['default']} |\n"
    
    doc += """
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
"""
    
    return doc


def main() -> None:
    """Generate and save the architecture documentation."""
    print("Generating architecture documentation...")
    
    doc = generate_architecture_doc()
    
    output_path = _project_root / "docs" / "ARCHITECTURE.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        f.write(doc)
    
    print(f"Saved architecture documentation to {output_path}")


if __name__ == "__main__":
    main()
