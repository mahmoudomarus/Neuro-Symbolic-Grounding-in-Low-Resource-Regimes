# NSCA: Neuro-Symbolic Cognitive Architecture

<p align="center">
  <img src="https://img.shields.io/badge/python-3.9%2B-blue" alt="Python 3.9+">
  <img src="https://img.shields.io/badge/pytorch-2.0%2B-red" alt="PyTorch 2.0+">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License">
</p>

A biologically-inspired cognitive architecture implementing a **layered approach to machine understanding**, combining innate perceptual priors with learned representations, causal reasoning, intrinsic motivation, and language grounding.

## Key Differentiators

Unlike conventional deep learning approaches that learn everything from scratch:

| Aspect | Traditional Models | NSCA |
|--------|-------------------|------|
| **Perception** | Learn pixels → concepts | Innate priors (color, edges, depth) + learning |
| **Understanding** | Pattern matching | Property-based semantics (hard, heavy, animate) |
| **Prediction** | Statistical correlation | Causal reasoning with intuitive physics |
| **Learning** | External reward only | Intrinsic motivation (curiosity, competence) |
| **Language** | Token prediction | Grounded in perceptual concepts |

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Layer 4: LANGUAGE INTEGRATION                                          │
│  ├─ Concept ↔ Word bidirectional grounding                              │
│  ├─ LLM integration for complex reasoning                               │
│  └─ Property-based verbalization                                        │
├─────────────────────────────────────────────────────────────────────────┤
│  Layer 3: MOTIVATION / DRIVE SYSTEM                                     │
│  ├─ Curiosity drive (seek novel, learnable experiences)                 │
│  ├─ Competence drive (seek mastery)                                     │
│  ├─ Intrinsic reward computation                                        │
│  └─ Attention allocation                                                │
├─────────────────────────────────────────────────────────────────────────┤
│  Layer 2: CAUSAL REASONING                                              │
│  ├─ Cause → Effect understanding via intervention                       │
│  ├─ Intuitive physics (gravity, solidity, support)                      │
│  └─ Counterfactual reasoning ("what if...")                             │
├─────────────────────────────────────────────────────────────────────────┤
│  Layer 1: SEMANTIC PROPERTIES                                           │
│  ├─ Physical properties (hardness, weight, size)                        │
│  ├─ Affordances (graspable, sittable, throwable)                        │
│  └─ Categories (agent/object, animate/inanimate)                        │
├─────────────────────────────────────────────────────────────────────────┤
│  Layer 0: WORLD MODEL                                                   │
│  ├─ Innate priors (color opponency, Gabor filters, mel-frequency)       │
│  ├─ Multi-modal encoders (vision, audio, proprioception)                │
│  ├─ Cross-modal fusion with attention                                   │
│  ├─ Temporal processing with causal masking                             │
│  ├─ Dynamics prediction (imagination engine)                            │
│  └─ Dual memory (episodic + semantic)                                   │
└─────────────────────────────────────────────────────────────────────────┘
```

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/NSCA.git
cd NSCA

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python verify_world_model.py
```

## Quick Start

### Using the Cognitive Agent

```python
import torch
from src.cognitive_agent import create_cognitive_agent

# Create agent with default configuration
agent = create_cognitive_agent(use_llm=False)

# Perceive multi-modal input
vision = torch.randn(1, 2, 3, 64, 64)  # [B, T, C, H, W]
audio = torch.randn(1, 16000)           # [B, samples]
proprio = torch.randn(1, 2, 12)         # [B, T, body_dim]

result = agent.perceive(vision, audio, proprio)

# Access results from all layers
print(f"World state: {result['world_state'].shape}")
print(f"Hardness: {result['properties'].hardness.item():.2f}")
print(f"Category: {result['category'].primary_category().value}")
print(f"Curiosity: {result['drive_state'].curiosity_level:.2f}")
print(f"Description: {result['description']}")

# Ask questions
answer = agent.what_is_this()
print(f"What is this: {answer['description']}")

# Imagine future states
actions = torch.randn(1, 3, 16)
imagination = agent.imagine(actions)
print(f"Predicted trajectory: {imagination['predicted_states'].shape}")
```

## Documentation

| Document | Description |
|----------|-------------|
| [Architecture](docs/ARCHITECTURE.md) | Detailed system architecture |
| [Theoretical Foundations](docs/THEORY.md) | Scientific basis and design rationale |
| [API Reference](docs/API.md) | Complete API documentation |
| [Training Guide](docs/TRAINING.md) | Training procedures and hyperparameters |
| [Paper](paper/manuscript.md) | Academic manuscript |

## Project Structure

```
NSCA/
├── src/
│   ├── cognitive_agent.py      # Unified cognitive architecture
│   │
│   ├── priors/                 # Innate perceptual priors
│   │   ├── visual_prior.py     # Color opponency, Gabor filters, depth cues
│   │   ├── audio_prior.py      # Mel-frequency, onset detection
│   │   ├── spatial_prior.py    # 3D spatial reasoning
│   │   └── temporal_prior.py   # Causal masking, temporal decay
│   │
│   ├── encoders/               # Multi-modal encoders
│   │   ├── vision_encoder.py   # CNN with integrated priors
│   │   ├── audio_encoder.py    # Cochlear-like processing
│   │   └── proprio_encoder.py  # Body state encoding
│   │
│   ├── fusion/                 # Cross-modal integration
│   │   └── cross_modal.py      # Attention-based fusion
│   │
│   ├── world_model/            # Layer 0: World representation
│   │   ├── temporal_world_model.py
│   │   ├── enhanced_dynamics.py
│   │   └── unified_world_model.py
│   │
│   ├── semantics/              # Layer 1: Semantic properties
│   │   ├── property_layer.py   # Property extraction
│   │   ├── affordances.py      # Action possibilities
│   │   └── categories.py       # Object categorization
│   │
│   ├── reasoning/              # Layer 2: Causal reasoning
│   │   ├── causal_layer.py     # Cause-effect learning
│   │   ├── intuitive_physics.py # Physics priors
│   │   └── counterfactual.py   # "What if" reasoning
│   │
│   ├── motivation/             # Layer 3: Drive system
│   │   ├── drive_system.py     # Curiosity, competence
│   │   ├── intrinsic_reward.py # Internal rewards
│   │   └── attention.py        # Focus allocation
│   │
│   ├── language/               # Layer 4: Language integration
│   │   └── llm_integration.py  # Concept-word grounding
│   │
│   ├── memory/                 # Memory systems
│   │   └── dual_memory.py      # Episodic + semantic
│   │
│   └── learning/               # Meta-learning
│       └── meta_learner.py     # MAML, prototypical networks
│
├── configs/                    # Configuration files
├── scripts/                    # Training scripts
├── tests/                      # Test suite
├── docs/                       # Documentation
└── paper/                      # Academic manuscript
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test modules
pytest tests/test_cognitive_layers.py -v
pytest tests/test_world_model.py -v

# Run verification script
python verify_world_model.py
```

## Training

```bash
# Train the world model
python scripts/train_world_model.py --config configs/training_config.yaml

# Monitor training
wandb login  # Optional: for experiment tracking
```

## Citation

If you use this work, please cite:

```bibtex
@software{nsca2026,
  author = {NSCA Contributors},
  title = {NSCA: Neuro-Symbolic Cognitive Architecture},
  year = {2026},
  url = {https://github.com/your-username/NSCA}
}
```

## Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) before submitting PRs.

## License

MIT License - see [LICENSE](LICENSE) for details.
