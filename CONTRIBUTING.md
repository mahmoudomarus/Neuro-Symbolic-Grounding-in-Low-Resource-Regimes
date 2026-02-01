# Contributing to NSCA

Thank you for your interest in contributing to the Neuro-Symbolic Cognitive Architecture project! This document provides guidelines for contributing.

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow

## How to Contribute

### Reporting Issues

1. Check if the issue already exists
2. Use the issue template
3. Include:
   - Clear description
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details (OS, Python version, PyTorch version)

### Submitting Pull Requests

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes
4. Run tests: `pytest tests/ -v`
5. Run linting: `flake8 src/`
6. Commit with clear messages
7. Push and create a PR

### Pull Request Guidelines

- One PR per feature/fix
- Include tests for new functionality
- Update documentation as needed
- Follow existing code style
- Keep PRs focused and small

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/NSCA.git
cd NSCA

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install dev dependencies
pip install pytest flake8 black mypy

# Run tests
pytest tests/ -v

# Run linting
flake8 src/
black --check src/
```

## Code Style

### Python Style

- Follow PEP 8
- Use type hints
- Write docstrings (Google style)
- Maximum line length: 100 characters

### Example

```python
def extract_properties(
    world_state: torch.Tensor,
    audio_features: Optional[torch.Tensor] = None,
) -> Tuple[PropertyVector, torch.Tensor]:
    """
    Extract semantic properties from world state.
    
    Args:
        world_state: World representation [B, state_dim]
        audio_features: Audio features [B, audio_dim] (optional)
        
    Returns:
        Tuple of:
        - PropertyVector with semantic properties
        - Property embedding [B, hidden_dim]
        
    Raises:
        ValueError: If world_state has wrong dimensions
    """
    if world_state.dim() != 2:
        raise ValueError(f"Expected 2D tensor, got {world_state.dim()}D")
    
    # Implementation...
```

### Commit Messages

Format: `<type>: <description>`

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `test`: Tests
- `refactor`: Code refactoring
- `style`: Formatting
- `perf`: Performance

Examples:
```
feat: add counterfactual reasoning module
fix: correct dimension mismatch in fusion layer
docs: update API documentation for property layer
test: add tests for causal reasoning
```

## Project Structure

```
NSCA/
├── src/                    # Source code
│   ├── cognitive_agent.py  # Main agent
│   ├── priors/            # Innate priors
│   ├── encoders/          # Multi-modal encoders
│   ├── semantics/         # Property extraction
│   ├── reasoning/         # Causal reasoning
│   ├── motivation/        # Drive system
│   └── language/          # Language integration
├── tests/                  # Test suite
├── docs/                   # Documentation
├── configs/               # Configuration files
└── scripts/               # Training scripts
```

## Testing

### Running Tests

```bash
# All tests
pytest tests/ -v

# Specific module
pytest tests/test_cognitive_layers.py -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

### Writing Tests

```python
import pytest
import torch

class TestPropertyLayer:
    """Tests for property extraction."""
    
    def test_property_extraction(self):
        """Properties can be extracted from world state."""
        from src.semantics.property_layer import PropertyLayer, PropertyConfig
        
        config = PropertyConfig(world_state_dim=64)
        layer = PropertyLayer(config)
        
        state = torch.randn(2, 64)
        props, embed = layer(state)
        
        assert props.hardness.shape == (2,)
        assert (props.hardness >= 0).all()
        assert (props.hardness <= 1).all()
```

## Documentation

### Docstrings

Use Google style docstrings:

```python
def function(arg1: int, arg2: str) -> bool:
    """
    Short description.
    
    Longer description if needed.
    
    Args:
        arg1: Description of arg1
        arg2: Description of arg2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When something is wrong
        
    Example:
        >>> function(1, "test")
        True
    """
```

### Documentation Files

- Update `docs/` when adding features
- Keep README.md current
- Add examples for new functionality

## Areas for Contribution

### High Priority

- [ ] Additional innate priors (e.g., biological motion)
- [ ] More intuitive physics rules
- [ ] Extended affordance set
- [ ] Better LLM integration

### Medium Priority

- [ ] Visualization tools
- [ ] Additional evaluation metrics
- [ ] Performance optimizations
- [ ] More training datasets

### Documentation

- [ ] Tutorials
- [ ] Example notebooks
- [ ] Video walkthroughs

## Questions?

- Open an issue for questions
- Tag with "question" label
- Check existing issues first

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
